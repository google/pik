// Copyright 2018 Google Inc. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//    http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//
// Disclaimer: This is not an official Google product.

#ifndef TILE_FLOW_H_
#define TILE_FLOW_H_

// Organizes image operations into a processing graph and executes them on tiles
// (small square regions). This leads to better cache utilization than
// processing entire images in each step and also enables parallel execution.
//
// Usage example: tile_flow_benchmark.cc.

#include <stddef.h>
#include <stdint.h>
#include <sys/types.h>
#include <algorithm>
#include <memory>  // std::unique_ptr
#include <string>
#include <utility>
#include <vector>

#include "bits.h"
#include "compiler_specific.h"
#include "data_parallel.h"
#include "image.h"
#include "status.h"

namespace pik {

struct OutputRegion {
  // Top left of the tile in the coordinate space of the (notional) materialized
  // output image. Only rarely needed, e.g. for custom sources; most nodes
  // instead only access their inputs starting at 0,0 or -borderX,-borderY.
  // Must be signed to support borders.
  int32_t x;
  int32_t y;

  // Dimensions of output buffer, i.e. tile_size * 2^scale + border. Serves as
  // loop bounds for node TFFunc.
  uint32_t xsize;
  uint32_t ysize;

  // Same as [x/y]size except for the bottom/right tiles, which are truncated to
  // the total image size for nodes/sinks.
  uint32_t partial_xsize;
  uint32_t partial_ysize;
};

// Number of border pixels per side. Nodes that compute their output using
// input pixels at a different location (e.g. convolution) require larger input
// sizes (e.g. expanding by the radius) to produce a given output size. Border
// sizes are usually the same on each side, but some operations such as causal
// predictors require more input on a particular side.
class Borders {
 public:
  // Top/bottom/left/right.
  enum Side { T, B, L, R, kNumSides };

  // Default: no border, output size matches the input.
  Borders() { std::fill(borders_, borders_ + kNumSides, 0); }

  // Same border size on each side (e.g. convolution radius).
  explicit Borders(const int border) {
    PIK_CHECK(border >= 0);
    std::fill(borders_, borders_ + kNumSides, border);
  }

  // Returns border size [number of pixels] for the given side.
  int operator[](const int side) const {
    PIK_CHECK(side < kNumSides);
    return borders_[side];
  }

  static Borders SumOf(const Borders& first, const Borders& second) {
    Borders sum;
    for (int side = 0; side < kNumSides; ++side) {
      sum.borders_[side] = first.borders_[side] + second.borders_[side];
    }
    return sum;
  }

  // Assigns a border size for the given side.
  void Set(const int side, const int border) {
    PIK_CHECK(side < kNumSides);
    PIK_CHECK(border >= 0);
    borders_[side] = border;
  }

  void UpdateToMax(const Borders& other) {
    for (int side = 0; side < kNumSides; ++side) {
      Set(side, std::max(borders_[side], other.borders_[side]));
    }
  }

  bool HasAny() const {
    for (int side = 0; side < kNumSides; ++side) {
      if (borders_[side] != 0) {
        return true;
      }
    }
    return false;
  }

  std::string ToString() const;

 private:
  int borders_[kNumSides];  // never negative.
};

// Scales (multiplies by 2^x) sizes of this node's outputs.
struct Scale {
  Scale() : shift_x(0), shift_y(0) {}
  Scale(const int x, const int y) : shift_x(x), shift_y(y) {}

  // Multiply by 2^x; positive = shift left, negative = right (divide).
  int shift_x;
  int shift_y;
};

// "Wrap modes" - how to initialize pixels that are outside the source image.
enum class TFWrap : uint32_t {
  // Set to zero (= OpenGL's CLAMP_TO_BORDER). Fast, default.
  kZero,

  // Mirror from valid pixels (= MIRRORED_REPEAT). Required for convolutions.
  kMirror
};

// All supported output types. A strongly-typed enum allows the compiler to
// verify that all cases are handled.
enum class TFType : uint32_t { kF32, kI32, kI16, kU16, kU8 };

struct TFTypeUtils {
  static inline TFType FromT(float) { return TFType::kF32; }
  static inline TFType FromT(int) { return TFType::kI32; }
  static inline TFType FromT(int16_t) { return TFType::kI16; }
  static inline TFType FromT(uint16_t) { return TFType::kU16; }
  static inline TFType FromT(uint8_t) { return TFType::kU8; }

  // For type-erasure in SetSource/Sink.
  template <typename T>
  static inline TFType FromImage(const Image<T>*) {
    return FromT(T());
  }
  template <typename T>
  static inline TFType FromImage(const Image3<T>*) {
    return FromT(T());
  }

  // Returns bytes per pixel; used to compute x offset in sinks.
  static inline size_t Size(const TFType type) {
    switch (type) {
      case TFType::kF32:
      case TFType::kI32:
        return 4;
      case TFType::kI16:
      case TFType::kU16:
        return 2;
      case TFType::kU8:
        return 1;
      default:
        return 0;
    }
  }

  // Returns string identifying the type; used in ToString.
  static inline const char* String(const TFType type) {
    switch (type) {
      case TFType::kF32:
        return "F";
      case TFType::kI32:
        return "I";
      case TFType::kI16:
        return "S";
      case TFType::kU16:
        return "U";
      case TFType::kU8:
        return "B";
      default:
        return "?";
    }
  }
};

// User-specified implementation of a node - fills output image[s] based on an
// optional extra argument and "inputs" (prior node output[s]). Note that
// "inputs" may point to the same buffer as "outputs" if the node is `in-place'.
// "output_region" indicates how many output pixels to produce.
//
// Function pointers are faster than virtual function calls and std::function.
// This can bind to a static function, captureless lambda with this signature,
// or a capturing lambda without the "arg" argument.
using TFFunc = void (*)(const void* arg,
                        const ConstImageViewF* PIK_RESTRICT inputs,
                        const OutputRegion& output_region,
                        const MutableImageViewF* PIK_RESTRICT outputs);

using TFPortBits = uint32_t;
using TFPortIndex = uint32_t;  // [0, kMaxPorts)

// Opaque pointer returned by TFBuilder::Add*. Only for use with TFPorts and
// the following accessors.
class TFNode;

// Returns 2^num_ports - 1.
TFPortBits AllPorts(const TFNode* node);
TFType OutType(const TFNode* node);

// A node and one or more of its output ports; used by TFBuilder::Add/Set to
// reference outputs of prior nodes, and bind to source/sink images.
struct TFPorts {
  // Allows implicit conversion for the common case of referring to all ports.
  TFPorts(TFNode* node) : node(node), bits(AllPorts(node)) {}

  // Requests a subset of ports.
  TFPorts(TFNode* node, const TFPortBits ports) : node(node), bits(ports) {
    // At least one port, and only available ports.
    PIK_CHECK(0 != bits && bits <= AllPorts(node));
  }

  bool IsSinglePort() const { return bits != 0 && (bits & (bits - 1)) == 0; }

  TFNode* const node;
  const TFPortBits bits;
};

// Breaks the circular dependency between TFGraph and TFBuilder.
class TFBuilderImpl;

// Compiled graph for a specific size/pool configuration. Thread-compatible.
class TFGraph {
 public:
  // Called by TFBuilder::Finalize, see documentation there.
  TFGraph(const ImageSize sink_size, const ImageSize tile_size,
          ThreadPool* pool, const TFBuilderImpl* builder);
  ~TFGraph();

  // Non-copyable because the dtor deletes instances_.
  TFGraph(TFGraph&&) = delete;
  TFGraph(const TFGraph&) = delete;
  TFGraph& operator=(TFGraph&&) = delete;
  TFGraph& operator=(const TFGraph&) = delete;

  // Runs the processing graph for every tile that overlaps sink_size - either
  // on the current thread or on worker threads. Can be called more than once,
  // or even concurrently across multiple instances.
  void Run();

 private:
  const uint32_t num_tiles_x_;
  const uint32_t num_tiles_y_;
  const uint32_t num_tiles_;

  ThreadPool* const pool_;                       // not owned.
  uint32_t num_instances_;                       // = pool_->NumThreads() or 1.
  uint8_t* instances_[ThreadPool::kMaxThreads];  // owned.
};

using TFGraphPtr = std::unique_ptr<TFGraph>;

// Builds a (directed) TFGraph by adding nodes and their dependencies.
class TFBuilder {
 public:
  TFBuilder();
  ~TFBuilder();

  // Adds a `source' with the given number of inputs. Checks that no source
  // is added after any other node (see Note below). Requires a subsequent call
  // to SetSource(port, source_image) for every port.
  TFNode* AddSource(const char* name, const size_t num_ports,
                    const TFType out_type, const TFWrap = TFWrap::kZero,
                    const Scale& out_scale = Scale());

  // Adds a `node', which may reference the outputs of sources or prior nodes.
  // A node can be changed to a `sink' by calling SetSink.
  //
  // "name" is useful for visualizing the graph.
  // "in_borders" is the number of additional input pixels on each side required
  //   for producing the OutputRegion.
  // "out_scale" are shift counts (positive or negative) to scale the output
  //   size by powers of two. This is preferable to scaling the _input_ sizes,
  //   which cannot handle nodes with different-sized inputs.
  // "inputs" are zero or more TFPorts, i.e. outputs from prior nodes that are
  //   required for computing the current node's output. Within "func",
  //   inputs[] provides access to these nodes' outputs in the same order.
  // "num_ports" indicates the number of output ports (>= 1) accessible via
  //   the outputs[] argument to "func". Subsequent calls to Add or SetSink may
  //   also reference this many ports from this node. Each output must have the
  //   same size (otherwise, use multiple nodes).
  // "func" is a TFFunc or equivalent captureless lambda that typically relays
  //   its arguments to another function. It writes pixels to outputs[] starting
  //   from outputs[o].Row(0), optionally referring to inputs[i].Row(y), where
  //   0 <= i < inputs.size(), 0 <= o < num_ports and
  //   -in_borders[Borders::T] <= y < out_ysize + in_borders[Borders::B].
  // "arg" (arg_size bytes) will be copied and passed to func. This is typically
  //   null (no argument needed) when called directly, because nodes that need
  //   extra arguments should use AddClosure instead (lambda captures are more
  //   convenient than manually unpacking from "arg"). Note that AddClosure
  //   calls this function with a non-null arg.
  //
  // The node's TFFunc may be called with `in-place' output if:
  //  - the node is not a sink (=> MutableImageViewF does not point to a sink
  //    ImageF) AND
  //  - inputs.size() >= num_ports (=> enough existing buffers) AND
  //  - in_borders == Borders() (=> no dependency on neighboring pixels) AND
  //  - no subsequent node references "inputs" (=> can reuse their outputs).
  // If so, TFFunc inputs[j] and outputs[j] point to the same MutableImageViewF,
  // so TFFunc must read from inputs[j] before writing to outputs[j].
  //
  // Note: the graph executes nodes in the order in which they were added.
  // To avoid type dispatch, all sources must precede other nodes, and sinks
  // must come last. The graph cannot reorder any nodes because they may have
  // side effects. The node order should maximize opportunities for in-place
  // operation (improves L1 hit rates), or at least reusing previous buffers
  // (reduces working set size).
  TFNode* Add(const char* name, const Borders& in_borders,
              const Scale& out_scale, std::vector<TFPorts>&& inputs,
              const size_t num_ports, const TFType out_type, const TFFunc func,
              const uint8_t* arg = nullptr, const size_t arg_size = 0);

  // Adds a node as above, but with a closure (lambda with capture list).
  // This is useful for relaying other arguments to the node's implementation.
  //
  // "closure" is a lambda with (const ConstImageViewF*, const OutputRegion&,
  // const MutableImageViewF*) arguments.
  template <class Closure>
  TFNode* AddClosure(const char* name, const Borders& in_borders,
                     const Scale& in_shifts, std::vector<TFPorts>&& inputs,
                     const size_t num_ports, const TFType out_type,
                     const Closure& closure) {
    return Add(name, in_borders, in_shifts, std::move(inputs), num_ports,
               out_type, &CallClosure<Closure>,
               reinterpret_cast<const uint8_t*>(&closure), sizeof(closure));
  }

  // Captures a pointer to a source image and binds it to the given port. "T"
  // can be any of TFType. Callers may resize the image before calling
  // Finalize. Finalize verifies this is called for each port in [0, num_ports).
  template <typename T>
  void SetSource(const TFPorts& ports, const Image<T>* source) {
    PIK_CHECK(ports.IsSinglePort());
    const TFPortIndex port = NumZeroBitsBelowLSBNonzero(ports.bits);
    BindSource(ports.node, port, reinterpret_cast<const ImageF*>(source),
               TFTypeUtils::FromImage(source));
  }

  // Binds the three source planes to the three specified ports.
  template <typename T>
  void SetSource(const TFPorts& ports, const Image3<T>* source) {
    TFPortBits bits = ports.bits;
    for (int c = 0; c < 3; ++c) {
      PIK_CHECK(bits != 0);
      const TFPortIndex port = NumZeroBitsBelowLSBNonzero(bits);
      bits &= bits - 1;
      BindSource(ports.node, port,
                 reinterpret_cast<const ImageF*>(&source->Plane(c)),
                 TFTypeUtils::FromImage(source));
    }
    PIK_CHECK(bits == 0);
  }

  // Converts the given node to a `sink', which writes directly to an output
  // image instead of allocating a buffer. Captures a pointer to the sink image
  // and binds it to the given port. Callers may resize the image before calling
  // Finalize. Finalize verifies this is called for each port, and ensures all
  // sink nodes were added after normal nodes.
  //
  // "sink" can have any type; TFFunc is responsible for casting row pointers to
  // the actual underlying type. The pointer-to-const allows binding to
  // &Image3::Plane(), and emphasizes that the image class will not be changed
  // (e.g. resized). However, the image *pixels* are of course modified.
  template <typename T>
  void SetSink(const TFPorts& ports, const Image<T>* sink) {
    PIK_CHECK(ports.IsSinglePort());
    const TFPortIndex port = NumZeroBitsBelowLSBNonzero(ports.bits);
    BindSink(ports.node, port, reinterpret_cast<const ImageF*>(sink),
             TFTypeUtils::FromImage(sink));
  }

  // Binds the three sink planes to the three specified ports.
  template <typename T>
  void SetSink(const TFPorts& ports, Image3<T>* sink) {
    const TFType type = TFTypeUtils::FromImage(sink);

    TFPortBits bits = ports.bits;
    for (int c = 0; c < 3; ++c) {
      PIK_CHECK(bits != 0);
      const TFPortIndex port = NumZeroBitsBelowLSBNonzero(bits);
      bits &= bits - 1;
      BindSink(ports.node, port,
               reinterpret_cast<const ImageF*>(&sink->Plane(c)), type);
    }
    PIK_CHECK(bits == 0);
  }

  // Returns a newly created, ready-to-run TFGraph. It is safe to destroy the
  // builder immediately afterwards, but this may also be called multiple times
  // (e.g. to associate the same nodes with a different ThreadPool).
  // No subsequent calls to Add* or SetSink are allowed.
  //
  // "sink_size" is the output size (> 0, need not be a multiple of tile_size).
  // "tile_size" is the size of an output tile (its dependencies may be larger
  //   due to Borders/Scale). It should be chosen such that all buffers fit into
  //   256 KiB L2 caches, and must be a power of two to allow bit-shifting.
  // "pool" can have NumThreads() == 0, in which case TFGraph::Run executes on
  //   the current thread. It must remain accessible for all calls to Run.
  TFGraphPtr Finalize(const ImageSize sink_size, const ImageSize tile_size,
                      ThreadPool* pool);

 private:
  // Type-erased implementations avoid multiple overloads.
  void BindSource(TFNode* node, TFPortIndex port, const ImageF* source,
                  TFType type);
  void BindSink(TFNode* node, TFPortIndex port, const ImageF* sink, TFType type);

  // Calls operator() of the closure (lambda function) "arg", which is a copy of
  // the original closure to improve locality. &CallClosure will be stored in a
  // TFFunc*, which has the same signature to avoid casting. Such a function
  // pointer is faster than std::function (no vtbl lookups/null checks).
  template <class Closure>
  static void CallClosure(const void* arg,
                          const ConstImageViewF* PIK_RESTRICT inputs,
                          const OutputRegion& output_region,
                          const MutableImageViewF* PIK_RESTRICT outputs) {
    (*static_cast<const Closure*>(arg))(inputs, output_region, outputs);
  }

  std::unique_ptr<TFBuilderImpl> impl_;
};

}  // namespace pik

#endif  // TILE_FLOW_H_
