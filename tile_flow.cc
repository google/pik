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

#include "tile_flow.h"

#include <atomic>
#include <cstdint>
#include <cstdio>
#include <cstring>

#define PROFILER_ENABLED 1
#include "arch_specific.h"
#include "cache_aligned.h"
#include "profiler.h"
#include "simd_helpers.h"

// Prints graph summary.
#define VERBOSE_GRAPH 1
// Prints per-node sizes etc.
#define VERBOSE_NODES 2
// Prints per-tile positions etc.
#define VERBOSE_TILES 4

#define VERBOSE 0

namespace pik {
namespace {

using TileIndex = uint32_t;  // <= imageSize / tileSize

using NodeBits = uint64_t;

// Upper bound on total TFBuilder::Add* calls. Used for vector<TFNode>.reserve
// to ensure TFNode pointers are never invalidated.
static constexpr size_t kMaxNodes = 32;

// Upper bound on TFBuilder::Add* num_ports. Determines SinkTLS array size.
static constexpr size_t kMaxPorts = 3;

static constexpr size_t kMaxSourceUsers = 2;

static_assert(kMaxPorts <= sizeof(TFPortBits) * 8, "Too many ports");
static_assert(kMaxNodes <= sizeof(NodeBits) * 8, "Too many nodes");

// Returns num/den, rounded up.
PIK_INLINE uint32_t CeilDiv(const uint32_t num, const uint32_t den) {
  return (num + den - 1) / den;
}

template <typename T>
PIK_INLINE T RoundUp(const T x, const int multiple_pow2) {
  return (x + multiple_pow2 - 1) & ~(multiple_pow2 - 1);
}

// Returns value * 2^count; count can be negative.
PIK_INLINE uint32_t ShiftBy(const uint32_t value, const int count) {
  return (count < 0) ? (value >> (-count)) : value << count;
}

// Converts tile index to pixel coordinates for source/sink. Used to compute
// OutputRegion::x/y, which may also be used by nodes. POD.
class CoordFromIndex {
 public:
  static CoordFromIndex FromTileSize(const uint32_t size) {
    CoordFromIndex ret;
    PIK_CHECK(size != 0);
    ret.shift_ = FloorLog2Nonzero(size);
    ret.border_ = 0;
    return ret;
  }

  void Add(const CoordFromIndex& prev, const int shift, const int border) {
    // Default-initialized => overwrite.
    if (shift_ == 0) {
      shift_ = prev.shift_ + shift;
      border_ = ShiftBy(prev.border_, shift) + border;
      PIK_CHECK(shift_ >= 0);
      PIK_CHECK(border_ >= 0);
    } else {
      // Reached already-initialized node through another input path => ensure
      // it has the same transform.
      PIK_CHECK(shift_ == prev.shift_ + shift);
      // Do allow updating (increasing) the border.
      border_ = std::max<int>(border_, ShiftBy(prev.border_, shift) + border);
    }
  }

  // For ToString
  int Shift() const { return shift_; }
  int Border() const { return border_; }

  // Returns coordinates within a node's output given a 0-based tile index.
  PIK_INLINE uint32_t operator()(const TileIndex index) const {
    return (index << shift_) - border_;
  }

 private:
  // How much to shift TileIndex. Sum of all Scale for this and user nodes,
  // including tile_size.
  int shift_ = 0;   // >= 0
  int border_ = 0;  // >= 0
};

// Stores the original and pre-clamped (partial) [x/y]size of a node. More
// compact than storing all 4 combinations of full/partial ImageSize. POD.
class FullAndPartial {
  static constexpr size_t kFull = 0;
  static constexpr size_t kPartial = 1;

 public:
  PIK_INLINE uint32_t Full() const { return sizes_[kFull]; }
  PIK_INLINE uint32_t Partial() const { return sizes_[kPartial]; }

  void SetFull(const uint32_t size) { sizes_[kFull] = size; }

  void SetPartialForOutSize(const uint32_t sink_size) {
    const uint32_t partial = sink_size % Full();
    sizes_[kPartial] = (partial == 0) ? Full() : partial;
  }

  PIK_INLINE uint32_t Get(const uint32_t is_partial) const {
    return sizes_[is_partial ? kPartial : kFull];
  }

 private:
  uint32_t sizes_[2];
};

// Fixed-size name storage (smaller than std::string, no dynamic allocation).
class NodeName {
 public:
  explicit NodeName(const char* name) {
    PIK_CHECK(std::strlen(name) <= kNameChars);
    std::strncpy(name_, name, kNameChars);
    name_[kNameChars] = '\0';
  }

  const char* Get() const { return name_; }

 private:
  static constexpr size_t kNameChars = 15;
  char name_[kNameChars + 1];  // plus \0
};

// Boolean flag with the capability to reset any number of instances to False in
// constant time. This requires an externally provided "epoch", a strictly
// monotonically increasing value that starts near zero and never reaches ~0u.
// The flag is True iff epoch_ matches epoch.
class AutoResetBool {
 public:
  bool Test(const uint32_t epoch) const { return epoch_ == epoch; }
  void Set(const uint32_t epoch) { epoch_ = epoch; }

 private:
  uint32_t epoch_ = ~0u;
};

// Each port of a source needs a reference to the source image, an output
// buffer, and references to all its users' inputs (so they can be patched in
// zero-copy mode). SourceTLS stores a variable-length array of these. POD.
#pragma pack(push, 1)
struct SourcePortTLS {
  void Init(const ImageF& source_image, uint8_t* buffer,
            size_t buffer_bytes_per_row) {
    source_.Init(source_image);

    for (size_t idx_user = 0; idx_user < kMaxSourceUsers; ++idx_user) {
      users_[idx_user].Init();
    }

    output_.Init(buffer, buffer_bytes_per_row);
  }

  // Stashes pointer to a user's view for later PointUsersAt*.
  void AddUserInput(ConstImageViewF& input, const uint32_t border_x,
                    const uint32_t border_y) {
    const intptr_t base = reinterpret_cast<intptr_t>(this);
    const intptr_t offset = reinterpret_cast<intptr_t>(&input) - base;
    PIK_CHECK(offset != 0);

    // Insert into the first free slot.
    for (size_t idx_user = 0; idx_user < kMaxSourceUsers; ++idx_user) {
      if (users_[idx_user].offset == 0) {
        users_[idx_user].Set(offset, border_x, border_y);
        return;
      }
    }

    PIK_CHECK(false);  // too many users
  }

  // Redirects all users' views to point to a window of the sources with
  // top-left coordinates x,y (including borders).
  PIK_INLINE void PointUsersAtSource(const uint32_t x, const uint32_t y,
                                     const uint32_t type_size) const {
    const intptr_t base = reinterpret_cast<intptr_t>(this);
    for (size_t idx_user = 0; idx_user < kMaxSourceUsers; ++idx_user) {
      const User& user = users_[idx_user];
      const int16_t offset = user.offset;
      if (offset == 0) return;
      ConstImageViewF* view = reinterpret_cast<ConstImageViewF*>(base + offset);
      view->Init(source_, x + user.border_x, y + user.border_y, type_size);
    }
  }

  // Redirects all users' views to point to our output buffer[s].
  void PointUsersAtOutput(const uint32_t type_size) const {
    const intptr_t base = reinterpret_cast<intptr_t>(this);
    for (size_t idx_user = 0; idx_user < kMaxSourceUsers; ++idx_user) {
      const User& user = users_[idx_user];
      const int16_t offset = user.offset;
      if (offset == 0) return;
      ConstImageViewF* view = reinterpret_cast<ConstImageViewF*>(base + offset);
      view->Init(output_, user.border_x, user.border_y, type_size);
    }
  }

  // The output buffer for this port.
  MutableImageViewF output_;

  // Points to top-left of source; SourceTLS will add x/y offsets.
  ConstImageViewF source_;

  struct User {
    void Init() {
      offset = 0;
      border_x = border_y = 0;
    }

    void Set(const intptr_t offset, const uint32_t border_x,
             const uint32_t border_y) {
      // Must fit in 16 bits (instances are only a few KiB large).
      PIK_CHECK(-32768 <= offset && offset <= 32767);
      this->offset = static_cast<int16_t>(offset);

      // Must fit in 8 bits (borders are typically much smaller).
      PIK_CHECK(border_x < 256 && border_y < 256);
      this->border_x = border_x;
      this->border_y = border_y;
    }

    // Byte offset of input view from SourcePortTLS.
    int16_t offset;
    // Offset from top-left of prior output, i.e. total user border.
    uint8_t border_x;
    uint8_t border_y;
  };

  // Offset is smaller than pointer and enables an efficient power-of-two size.
  User users_[kMaxSourceUsers];
};
#pragma pack(pop)
static_assert(sizeof(SourcePortTLS) == 32, "Check size, 2^n more efficient");

// Stores pointers to the output buffers residing in this node's TLS; allows
// user nodes to retrieve them for their inputs. WARNING: single instance per
// node, so CreateInstance must not be called concurrently for the same graph.
struct OutputPointers {
  void Clear() {
    source_ports = nullptr;
    outputs = nullptr;
  }

  const MutableImageViewF& Get(const TFPortIndex idx_port) const {
    PIK_CHECK(IsSource() || outputs != nullptr);
    return source_ports ? source_ports[idx_port].output_ : outputs[idx_port];
  }

  void SetSource(SourcePortTLS* ports) {
    PIK_CHECK(!IsSource());
    PIK_CHECK(outputs == nullptr);
    source_ports = ports;
    PIK_CHECK(IsSource());
  }

  void SetOutputs(MutableImageViewF* outputs) {
    PIK_CHECK(!IsSource());
    PIK_CHECK(this->outputs == nullptr);
    this->outputs = outputs;
    PIK_CHECK(!IsSource());
  }

  bool IsSource() const { return source_ports != nullptr; }

  // Add a subsequent node's "input" view so it can be updated by the source.
  void AddUserInput(const TFPortIndex idx_port, ConstImageViewF& input,
                    const uint32_t border_x, const uint32_t border_y) {
    // Only needed/possible if we are a source.
    if (IsSource()) {
      source_ports[idx_port].AddUserInput(input, border_x, border_y);
    }
  }

  // Source nodes: points to SourceTLS.ports_, else null. Must be special-cased
  // because SourceTLS outputs are not contiguous. Also used by AddUserInput.
  SourcePortTLS* source_ports = nullptr;

  // Points to NodeTLS/SinkTLS.outputs, or null if source_ports != null.
  MutableImageViewF* outputs = nullptr;
};

}  // namespace

// Graph node of any type. Forward-declared in header => not in anonymous
// namespace. Used in four distinct phases:
// 1) graph construction: some fields already set by ctor; may add additional
//    users to existing nodes or bind them to sources/sinks.
// 2) backward pass: InitR traverses from sinks to sources, updating sizes.
// 3) forward pass: Finalize allocates the buffers.
// 4) compilation: copies fields to contiguous TLS. (single-threaded!)
class TFNode {
 public:
  TFNode() : index_(~0u), name_("?") {}

  // A source with "num_ports" inputs (images) and outputs (buffers).
  TFNode(const uint32_t index, const char* name, const size_t num_ports,
         TFType out_type, const TFWrap wrap, const Scale& out_scale)
      : index_(index),
        name_(name),
        out_scale_(out_scale),
        out_type_(out_type),
        source_wrap_(wrap),
        ports_(num_ports),
        is_source_(true) {
    PIK_CHECK(num_ports != 0);
    PIK_CHECK(num_ports <= kMaxPorts);
    PIK_CHECK(IsSource());
  }

  // A node or future sink that may read from ConstImageViewF[] and write to
  // MutableImageViewF[].
  TFNode(const uint32_t index, const char* name, const Borders& in_borders,
         const Scale& out_scale, std::vector<TFPorts>&& inputs,
         const size_t num_ports, const TFType out_type, const TFFunc func,
         const uint8_t* func_arg, const size_t func_arg_size)
      : index_(index),
        name_(name),
        in_borders_(in_borders),
        out_scale_(out_scale),
        inputs_(std::move(inputs)),
        out_type_(out_type),
        func_(func),
        func_arg_(RoundUp(func_arg_size, 8)),
        ports_(num_ports) {
    PIK_CHECK(num_ports != 0);
    PIK_CHECK(num_ports <= kMaxPorts);

    for (const TFPorts& input : inputs_) {
      // InitR requires inputs to be unique.
      PIK_CHECK(!input.node->already_used_as_input_.Test(index));
      input.node->already_used_as_input_.Set(index);

      num_inputs_ += PopCount(input.bits);

      // Indicate this node is a user of all specified input ports.
      for (TFPortBits bits = input.bits; bits != 0; bits &= bits - 1) {
        const TFPortIndex port = NumZeroBitsBelowLSBNonzero(bits);
        input.node->ports_[port].users |= 1ULL << Index();
      }
    }

    if (func_arg != nullptr) {
      memcpy(func_arg_.data(), func_arg, func_arg_size);
    }

    PIK_CHECK(!IsSource());
    PIK_CHECK(!IsSink());  // only true after BindSink
  }

  // Immediately valid and unchanging after ctor:

  uint32_t Index() const { return index_; }
  size_t NumInputs() const { return num_inputs_; }
  size_t NumPorts() const { return ports_.size(); }
  TFType OutType() const { return out_type_; }
  size_t ArgSize() const { return func_arg_.size(); }

  // Called during graph construction:

  bool IsSource() const { return is_source_; }
  bool IsSink() const { return is_sink_; }

  void BindImageImpl(const TFPortIndex idx_port, const ImageF* image,
                     const TFType type) {
    PIK_CHECK(idx_port < NumPorts());
    PIK_CHECK(image != nullptr);
    // NOTE: callers may reallocate/resize "source" until TFBuilder::Finalize,
    // so only store the pointer and do not access its fields.

    // Overwriting an existing port is probably a bug.
    PIK_CHECK(ports_[idx_port].image == nullptr);
    ports_[idx_port].image = image;

    // Must match reported output type of the source/TFFunc.
    PIK_CHECK(type == out_type_);
  }

  void BindSource(const TFPortIndex idx_port, const ImageF* source,
                  const TFType type) {
    PIK_CHECK(IsSource() && !IsSink());
    BindImageImpl(idx_port, source, type);
  }

  void BindSink(const TFPortIndex idx_port, const ImageF* sink,
                const TFType type) {
    PIK_CHECK(!IsSource());
    BindImageImpl(idx_port, sink, type);
    is_sink_ = true;
  }

  // Graph finalization:

  // Recursive init (from sink to source), called during TFBuilder::Finalize.
  // The next* arguments are from the "user" node that specified this node as an
  // input. Also called with user == this for each node.
  void InitR(const uint32_t num_finalize, const TFNode* user,
             const Borders& next_in_borders,
             const CoordFromIndex& next_x_from_ix,
             const CoordFromIndex& next_y_from_iy, const uint32_t next_xsize,
             const uint32_t next_ysize, const uint32_t next_sink_xsize,
             const uint32_t next_sink_ysize) {
    // Non-recursive call but already reached this node through a user's
    // recursive call - skip, the initial next_* args won't change anything.
    // Automatically resets to false after the next call to Finalize, which
    // allows us to update our sink size etc.
    if (user == this && init_epoch_.Test(num_finalize)) return;
    init_epoch_.Set(num_finalize);

    max_user_borders_.UpdateToMax(next_in_borders);

    x_from_ix_.Add(next_x_from_ix, out_scale_.shift_x,
                   max_user_borders_[Borders::L]);
    y_from_iy_.Add(next_y_from_iy, out_scale_.shift_y,
                   max_user_borders_[Borders::T]);
    total_scale_x_ += out_scale_.shift_x;
    total_scale_y_ += out_scale_.shift_y;

    out_xsizes_.SetFull(ComputeXSize(next_xsize));
    out_ysizes_.SetFull(ComputeYSize(next_ysize));
    sink_xsize_ = ShiftBy(next_sink_xsize, total_scale_x_);
    sink_ysize_ = ShiftBy(next_sink_ysize, total_scale_y_);
    out_xsizes_.SetPartialForOutSize(sink_xsize_);
    out_ysizes_.SetPartialForOutSize(sink_ysize_);

    for (const TFPorts& input : inputs_) {
      input.node->InitR(num_finalize, this, in_borders_, x_from_ix_, y_from_iy_,
                        out_xsizes_.Full(), out_ysizes_.Full(), sink_xsize_,
                        sink_ysize_);
    }
  }

  // Called by TFBuilder::Finalize in first to last order.
  void Finalize() {
    // Source or node: compute buffer size.
    if (!IsSink()) {
      const size_t type_size = TFTypeUtils::Size(out_type_);
      // Same logic as Image<T> - allows reading/writing an extra vector.
      buffer_bytes_per_row_ =
          BytesPerRow<kMaxVectorSize>(out_xsizes_.Full() * type_size);
    }
  }

  // Valid after Finalize, which is called once per TFBuilder::Finalize:

  CoordFromIndex GetXFromIndex() const { return x_from_ix_; }
  CoordFromIndex GetYFromIndex() const { return y_from_iy_; }
  FullAndPartial OutXSizes() const { return out_xsizes_; }
  FullAndPartial OutYSizes() const { return out_ysizes_; }

  // Returns size [bytes] of ONE port's buffer.
  size_t BufferSize() const {
    return buffer_bytes_per_row_ * out_ysizes_.Full();
  }

  size_t TotalBufferSize() const { return BufferSize() * NumPorts(); }

  std::string ToString() const {
    const std::string num_in = std::to_string(NumInputs());
    const std::string num_out = std::to_string(NumPorts());
    const std::string out_type(TFTypeUtils::String(out_type_));
    std::string type;

    if (IsSource()) {
      PIK_CHECK(NumInputs() == 0);
      type = std::string("Source") + out_type + ">" + num_out;
    } else if (IsSink()) {
      type = num_in + ">Sink" + out_type + ">" + num_out;
    } else {
      type = num_in + ">Node" + out_type + ">" + num_out;
    }

    char buf[200];
    std::snprintf(buf, sizeof(buf),
                  "%15s (%2d): %12s; in %s; max_user %s; "           // borders
                  "%3d x %3d (partial %3d x %3d, sink %5d x %3d); "  // sizes
                  "scale %2d %2d; CoordFromIndex <%2d %2d +%d %d",
                  name_.Get(), Index(), type.c_str(),
                  in_borders_.ToString().c_str(),
                  max_user_borders_.ToString().c_str(), out_xsizes_.Full(),
                  out_ysizes_.Full(), out_xsizes_.Partial(),
                  out_ysizes_.Partial(), sink_xsize_, sink_ysize_,
                  out_scale_.shift_x, out_scale_.shift_y, x_from_ix_.Shift(),
                  y_from_iy_.Shift(), x_from_ix_.Border(), y_from_iy_.Border());
    return buf;
  }

  // Copies the argument (typically closure) to arg.
  TFFunc GetFunc(uint8_t* func_arg) const {
    PIK_ASSERT(!IsSource());
    PIK_ASSERT(func_ != nullptr);
    memcpy(func_arg, func_arg_.data(), func_arg_.size());
    PIK_CHECK(uintptr_t(func_arg) % 8 == 0);
    return func_;
  }

  uint8_t* GetSources(uint8_t* buffers, SourcePortTLS* ports,
                      ImageSize* PIK_RESTRICT size, TFType* PIK_RESTRICT type,
                      TFWrap* PIK_RESTRICT wrap) const {
    PIK_CHECK(IsSource() && !IsSink());
    PIK_CHECK(NumPorts() != 0);  // ensure ports_[0] is accessible.

    *size = ImageSize::Make(ports_[0].image->xsize(), ports_[0].image->ysize());

    output_pointers_.SetSource(ports);

    for (TFPortIndex idx_port = 0; idx_port < NumPorts(); ++idx_port) {
      const Port& port = ports_[idx_port];
      SourcePortTLS& port_tls = ports[idx_port];

      // Size should match.
      PIK_CHECK(port.image != nullptr);  // else BindSource wasn't called
      PIK_CHECK(size->xsize == port.image->xsize());
      PIK_CHECK(size->ysize == port.image->ysize());

      port_tls.Init(*port.image, buffers, buffer_bytes_per_row_);
      buffers += BufferSize();  // nonzero
    }

    *type = out_type_;
    *wrap = source_wrap_;

    return buffers;
  }

  // Returns number of sinks (i.e. ports).
  size_t GetSinks(MutableImageViewF* PIK_RESTRICT outputs,
                  uint32_t* PIK_RESTRICT type_size) const {
    PIK_CHECK(!IsSource() && IsSink());

    for (TFPortIndex idx_port = 0; idx_port < NumPorts(); ++idx_port) {
      const ImageF* sink = ports_[idx_port].image;
      // Ensure BindSink was called for this port.
      PIK_CHECK(sink != nullptr);

      // Size must match - we can only pass a single OutputRegion to TFFunc.
      PIK_CHECK(sink_xsize_ == sink->xsize());
      PIK_CHECK(sink_ysize_ == sink->ysize());

      // Initially top-left of sink; SinkTLS.Run will move the output window.
      outputs[idx_port].Init(const_cast<uint8_t*>(sink->bytes()),
                             sink->bytes_per_row());
    }

    *type_size = TFTypeUtils::Size(out_type_);

    return NumPorts();
  }

  // Called by CreateInstance to ensure TLS pointers aren't stale.
  void ResetTLS() const { output_pointers_.Clear(); }

  void GetInputs(ConstImageViewF* PIK_RESTRICT inputs_tls) const {
    PIK_CHECK(!IsSource());

    size_t idx_input_tls = 0;
    for (const TFPorts input : inputs_) {
      const TFNode* prev = input.node;
      // TODO(janwas): allow reusing sink output?
      PIK_CHECK(!prev->IsSink());
      PIK_CHECK(prev->IsSource() == (prev->output_pointers_.IsSource()));
      const uint32_t border_x = prev->max_user_borders_[Borders::L];
      const uint32_t border_y = prev->max_user_borders_[Borders::T];

      // Foreach port, lowest to highest:
      for (TFPortBits bits = input.bits; bits != 0; bits &= bits - 1) {
        const TFPortIndex idx_port = NumZeroBitsBelowLSBNonzero(bits);
        ConstImageViewF& input_tls = inputs_tls[idx_input_tls++];

        input_tls.Init(prev->output_pointers_.Get(idx_port), border_x,
                       border_y);

        // Add ourselves as a user of prev.
        prev->output_pointers_.AddUserInput(idx_port, input_tls, border_x,
                                            border_y);
      }
    }
  }

  // Returns the next "buffers" (possibly unchanged if in-place/reusing).
  uint8_t* GetOutputs(uint8_t* buffers,
                      MutableImageViewF* PIK_RESTRICT outputs) const {
    PIK_CHECK(!IsSink());

    output_pointers_.SetOutputs(outputs);

    for (TFPortIndex idx_port = 0; idx_port < NumPorts(); ++idx_port) {
      outputs[idx_port].Init(buffers, buffer_bytes_per_row_);
      buffers += BufferSize();  // nonzero
    }

    return buffers;
  }

 private:
  uint32_t ComputeXSize(const uint32_t next_xsize) const {
    return max_user_borders_[Borders::L] +
           ShiftBy(next_xsize, out_scale_.shift_x) +
           max_user_borders_[Borders::R];
  }

  uint32_t ComputeYSize(const uint32_t next_ysize) const {
    return max_user_borders_[Borders::T] +
           ShiftBy(next_ysize, out_scale_.shift_y) +
           max_user_borders_[Borders::B];
  }

  // One instance per output of any kind of TFNode.
  struct Port {
    NodeBits users = 0;

    // For source/sink only; overwritten once by BindSource/BindSink.
    const ImageF* image = nullptr;
  };

  // "const", only set by ctor:

  uint32_t index_;
  NodeName name_;       // For ToString.

  Borders in_borders_;  // For InitR.
  Scale out_scale_;     // For InitR.

  std::vector<TFPorts> inputs_;
  size_t num_inputs_ = 0;  // sum of PopCnt(p.bits) for p in inputs_.
  AutoResetBool already_used_as_input_;  // for detecting duplicate inputs.

  TFType out_type_;
  TFWrap source_wrap_;

  TFFunc func_ = nullptr;
  std::vector<uint8_t> func_arg_;  // copied from a capturing lambda, or empty.

  // Updated by SetSink/Source:

  std::vector<Port> ports_;

  bool is_source_ = false;
  bool is_sink_ = false;

  // Updated by InitR:

  AutoResetBool init_epoch_;  // skips redundant calls to InitR.
  Borders max_user_borders_;
  CoordFromIndex x_from_ix_;  // = sum of InitR x_from_ix.
  CoordFromIndex y_from_iy_;
  int total_scale_x_ = 0;  // Used to compute sink_size; CoordFromIndex includes
  int total_scale_y_ = 0;  // tile_size and borders but these do not.
  FullAndPartial out_xsizes_;
  FullAndPartial out_ysizes_;
  uint32_t sink_xsize_;  // subject to this node's scale.
  uint32_t sink_ysize_;

  // Set by Finalize:

  size_t buffer_bytes_per_row_ = 0;

  // Updated during CreateInstance by ResetTLS:

  // Mutable so functions can be const to prevent modifying other fields.
  mutable OutputPointers output_pointers_;
};

namespace {

// To reduce cache pollution, we provide a compact graph representation which
// does not occupy much of L1 (< 2 KiB). We precompute as much as possible and
// choose smaller types, but not int16 because those are slower to read.

// Each worker thread needs its own tile buffers. The graph is also per-thread
// because node inputs may point to private buffers or shared images (which
// require shifting the input position for each tile), and patching pointers at
// runtime is about as costly as just initializing a per-thread graph.

// *TLS are POD to ensure their first 8 bytes differ from Sentinels::kMagic.
// It is also convenient to return a pointer from their Init 'constructors'.

// Passed to *TLS::Run (avoids re-capturing multiple times for lambdas)
struct RunArg {
  RunArg(const TileIndex ix, const TileIndex iy, const uint32_t num_x,
         const uint32_t num_y)
      : tile_ix(ix),
        tile_iy(iy),
        is_partial_x(ix == num_x - 1),
        is_partial_y(iy == num_y - 1) {}

  // Zero-based index of the tile.
  TileIndex tile_ix;
  TileIndex tile_iy;

  // For FullAndPartial::Get.
  uint32_t is_partial_x;
  uint32_t is_partial_y;
};

// Variable-size POD.
class SourceTLS {
 public:
  // Returns "buffers" for the next node.
  uint8_t* Init(const TFNode& node, uint8_t* buffers) {
    x_from_ix = node.GetXFromIndex();
    y_from_iy = node.GetYFromIndex();
    num_ports_ = node.NumPorts();
    buffers = node.GetSources(buffers, ports_, &source_size_, &type_, &wrap_);
    source_type_size_ = TFTypeUtils::Size(type_);

    const uintptr_t end = reinterpret_cast<uintptr_t>(ports_ + num_ports_);
    PIK_CHECK(end - reinterpret_cast<uintptr_t>(this) == Size(node));

    out_xsize_ = node.OutXSizes().Full();
    out_ysize_ = node.OutYSizes().Full();

    return buffers;
  }

  static size_t Size(const TFNode& node) {
    size_t size = sizeof(SourceTLS);
    size += (node.NumPorts() - 1) * sizeof(SourcePortTLS);
    PIK_CHECK(size % 8 == 0);
    return size;
  }

  PIK_INLINE SourceTLS* Next() {
    return reinterpret_cast<SourceTLS*>(ports_ + num_ports_);
  }
  PIK_INLINE const SourceTLS* Next() const {
    return reinterpret_cast<const SourceTLS*>(ports_ + num_ports_);
  }

  PIK_INLINE void Run(const RunArg& arg) const {
    // Top left of the tile in the source image.
    const int32_t x = x_from_ix(arg.tile_ix);
    const int32_t y = y_from_iy(arg.tile_iy);

    bool allow_copy = (x >= 0 && y >= 0);
    allow_copy &= (x + out_xsize_ <= source_size_.xsize);
    allow_copy &= (y + out_ysize_ <= source_size_.ysize);
    if (PIK_LIKELY(allow_copy)) {
      for (TFPortIndex idx_port = 0; idx_port < num_ports_; ++idx_port) {
        ports_[idx_port].PointUsersAtSource(x, y, source_type_size_);
      }

#if VERBOSE & VERBOSE_TILES
      printf("src zc %5d %5d size %3d x %3d (limit   %4d x %3d) myout %p\n", x,
             y, out_xsize_, out_ysize_, source_size_.xsize, source_size_.ysize,
             ports_[0].output_.ConstRow(0));
#endif
      return;
    }

#if VERBOSE & VERBOSE_TILES
    printf("src cp %5d %5d size %3d x %3d (limit   %4d x %3d)\n", x, y,
           out_xsize_, out_ysize_, source_size_.xsize, source_size_.ysize);
#endif
    PROFILER_ZONE("|| TFGraph RunSource slow");

    for (TFPortIndex idx_port = 0; idx_port < num_ports_; ++idx_port) {
      ports_[idx_port].PointUsersAtOutput(source_type_size_);
    }

    const OutputRegion output_region = {x,          y,          out_xsize_,
                                        out_ysize_, out_xsize_, out_ysize_};
    if (PIK_LIKELY(wrap_ == TFWrap::kZero)) {
      switch (type_) {
        case TFType::kF32:
          return RunT<float>(output_region);
        case TFType::kI32:
          return RunT<int>(output_region);
        case TFType::kI16:
          return RunT<int16_t>(output_region);
        case TFType::kU16:
          return RunT<uint16_t>(output_region);
        case TFType::kU8:
          return RunT<uint8_t>(output_region);
      }
    } else {
      switch (type_) {
        case TFType::kF32:
          return RunMirrorT<float>(output_region);
        case TFType::kI32:
          return RunMirrorT<int>(output_region);
        case TFType::kI16:
          return RunMirrorT<int16_t>(output_region);
        case TFType::kU16:
          return RunMirrorT<uint16_t>(output_region);
        case TFType::kU8:
          return RunMirrorT<uint8_t>(output_region);
      }
    }
  }

 private:
  template <typename T>
  PIK_INLINE void RunMirrorT(const OutputRegion& output_region) const {
    for (TFPortIndex idx_port = 0; idx_port < num_ports_; ++idx_port) {
      const SourcePortTLS& port = ports_[idx_port];

      // PrefetchNTA should reduce cache pollution (at least in L3), but
      // prefetching 256 bytes from 1..3 rows ahead does not yield an
      // appreciable improvement even with 8 threads and 1.5K x 1.5K images.

      for (int64_t oy = 0; oy < output_region.ysize; ++oy) {
        const int64_t iy = Mirror(output_region.y + oy, source_size_.ysize);
        const T* const PIK_RESTRICT row_in =
            reinterpret_cast<const T*>(port.source_.ConstRow(iy));
        T* PIK_RESTRICT row_out = reinterpret_cast<T*>(port.output_.Row(oy));

        for (int64_t ox = 0; ox < output_region.xsize; ++ox) {
          const int64_t ix = Mirror(output_region.x + ox, source_size_.xsize);
          row_out[ox] = row_in[ix];
        }
      }
    }
  }

  template <typename T>
  PIK_INLINE void RunT(const OutputRegion& output_region) const {
    using namespace SIMD_NAMESPACE;

    const int64_t ix_size = source_size_.xsize;
    const int64_t iy_size = source_size_.ysize;
    const int64_t ox_size = output_region.xsize;
    const int64_t oy_size = output_region.ysize;

    const int64_t ix0 = output_region.x;
    const int64_t iy0 = output_region.y;

    const int64_t zero = 0;
    const int64_t oy_zero = std::min(std::max(zero, -iy0), oy_size);
    const int64_t iy_remainder = iy_size - std::max(zero, iy0);
    const int64_t oy_end = std::min(oy_zero + iy_remainder, oy_size);

    const int64_t ox_zero = std::min(std::max(zero, -ix0), ox_size);
    const int64_t ix_remainder = ix_size - std::max(zero, ix0);
    const int64_t ox_end = std::min(ox_zero + ix_remainder, ox_size);

    // Currently, out of bounds = black. TODO(janwas): clamp/mirror?

    for (TFPortIndex idx_port = 0; idx_port < num_ports_; ++idx_port) {
      const SourcePortTLS& port = ports_[idx_port];

      int64_t oy = 0;
      for (; oy < oy_zero; ++oy) {
        float* PIK_RESTRICT row_out = port.output_.Row(oy);
        for (int64_t ox = 0; ox < ox_size; ++ox) {
          row_out[ox] = T(0);
        }
      }

      // PrefetchNTA should reduce cache pollution (at least in L3), but
      // prefetching 256 bytes from 1..3 rows ahead does not yield an
      // appreciable improvement even with 8 threads and 1.5K x 1.5K images.

      for (; oy < oy_end; ++oy) {
        const T* const PIK_RESTRICT row_in =
            reinterpret_cast<const T*>(port.source_.ConstRow(oy + iy0)) + ix0;
        T* PIK_RESTRICT row_out = reinterpret_cast<T*>(port.output_.Row(oy));

        int64_t ox = 0;
        for (; ox < ox_zero; ++ox) {
          row_out[ox] = T(0);
        }

        for (; ox < ox_end; ++ox) {
          row_out[ox] = row_in[ox];
        }

        for (; ox < ox_size; ++ox) {
          row_out[ox] = T(0);
        }
      }

      for (; oy < oy_size; ++oy) {
        T* PIK_RESTRICT row_out = reinterpret_cast<T*>(port.output_.Row(oy));
        for (int64_t ox = 0; ox < ox_size; ++ox) {
          row_out[ox] = T(0);
        }
      }
    }
  }

  CoordFromIndex x_from_ix;  // (Never matches Sentinels::kMagic)
  CoordFromIndex y_from_iy;
  uint32_t num_ports_;
  TFType type_;
  TFWrap wrap_;
  uint32_t source_type_size_;
  ImageSize source_size_;
  // No FullAndPartial because sources are never clamped - RunT takes care of
  // bounds-checking.
  uint32_t out_xsize_;
  uint32_t out_ysize_;
  SourcePortTLS ports_[1];  // num_ports_ entries (>= 1)
};

// Variable-size POD.
class NodeTLS {
 public:
  // Returns "buffers" for the next node.
  uint8_t* Init(const TFNode& node, uint8_t* buffers) {
    ConstImageViewF* PIK_RESTRICT inputs =
        reinterpret_cast<ConstImageViewF*>(this + 1);
    num_inputs_ = node.NumInputs();
    MutableImageViewF* outputs =
        reinterpret_cast<MutableImageViewF*>(inputs + num_inputs_);
    uint8_t* func_arg = reinterpret_cast<uint8_t*>(outputs + node.NumPorts());
    uint8_t* this_u8 = reinterpret_cast<uint8_t*>(this);
    func_arg_offset_ = RoundUp(func_arg - this_u8, 8);
    func_arg = this_u8 + func_arg_offset_;
    const size_t size = func_arg_offset_ + node.ArgSize();
    PIK_CHECK(size == Size(node));
    end_ = reinterpret_cast<NodeTLS*>(this_u8 + size);

    func_ = node.GetFunc(func_arg);
    x_from_ix = node.GetXFromIndex();
    y_from_iy = node.GetYFromIndex();
    node.GetInputs(inputs);
    out_xsizes_ = node.OutXSizes();
    out_ysizes_ = node.OutYSizes();
    return node.GetOutputs(buffers, outputs);
  }

  static size_t Size(const TFNode& node) {
    size_t size = sizeof(NodeTLS);
    size += node.NumInputs() * sizeof(ConstImageViewF);
    size += node.NumPorts() * sizeof(MutableImageViewF);
    size = RoundUp(size, 8);  // func_arg_offset_ is aligned.
    size += node.ArgSize();
    PIK_CHECK(size % 8 == 0);
    return size;
  }

  PIK_INLINE NodeTLS* Next() { return end_; }
  PIK_INLINE const NodeTLS* Next() const { return end_; }

  PIK_INLINE void Run(const RunArg& arg) const {
    const int32_t x = x_from_ix(arg.tile_ix);
    const int32_t y = y_from_iy(arg.tile_iy);

    const uint8_t* func_arg =
        reinterpret_cast<const uint8_t*>(this) + func_arg_offset_;
    const ConstImageViewF* inputs =
        reinterpret_cast<const ConstImageViewF*>(this + 1);
    const MutableImageViewF* PIK_RESTRICT outputs =
        reinterpret_cast<const MutableImageViewF*>(inputs + num_inputs_);
    const OutputRegion output_region = {x,
                                        y,
                                        out_xsizes_.Full(),
                                        out_ysizes_.Full(),
                                        out_xsizes_.Get(arg.is_partial_x),
                                        out_ysizes_.Get(arg.is_partial_y)};
#if VERBOSE & VERBOSE_TILES
    printf("node   %5d %5d size %3d x %3d (partial %3d x %3d) in %p out %p\n",
           x, y, output_region.xsize, output_region.ysize,
           output_region.partial_xsize, output_region.partial_ysize,
           inputs[0].ConstRow(0), outputs[0].ConstRow(0));
#endif

    func_(func_arg, inputs, output_region, outputs);
  }

 private:
  uint32_t num_inputs_;  // (Never matches Sentinels::kMagic)
  uint32_t func_arg_offset_;

  NodeTLS* end_;

  TFFunc func_;
  FullAndPartial out_xsizes_;
  FullAndPartial out_ysizes_;
  CoordFromIndex x_from_ix;
  CoordFromIndex y_from_iy;

  // Followed by: inputs, outputs, func_arg <- end
};

// Variable-size POD. Similar to NodeTLS but also updates sink image positions
// for every tile.
class SinkTLS {
 public:
  void Init(const TFNode& node) {
    ConstImageViewF* PIK_RESTRICT inputs =
        reinterpret_cast<ConstImageViewF*>(this + 1);
    num_inputs_ = node.NumInputs();
    MutableImageViewF* PIK_RESTRICT outputs =
        reinterpret_cast<MutableImageViewF*>(inputs + num_inputs_);
    uint8_t* func_arg = reinterpret_cast<uint8_t*>(outputs + node.NumPorts());
    uint8_t* this_u8 = reinterpret_cast<uint8_t*>(this);
    func_arg_offset_ = RoundUp(func_arg - this_u8, 8);
    func_arg = this_u8 + func_arg_offset_;
    const size_t size = func_arg_offset_ + node.ArgSize();
    PIK_CHECK(size == Size(node));
    end_ = reinterpret_cast<SinkTLS*>(this_u8 + size);

    func_ = node.GetFunc(func_arg);
    node.GetInputs(inputs);
    num_sinks_ = node.GetSinks(outputs, &type_size_);
    out_xsizes_ = node.OutXSizes();
    out_ysizes_ = node.OutYSizes();
    x_from_ix = node.GetXFromIndex();
    y_from_iy = node.GetYFromIndex();
    PIK_CHECK(num_sinks_ != 0);
  }

  static size_t Size(const TFNode& node) {
    size_t size = sizeof(SinkTLS);
    size += node.NumInputs() * sizeof(ConstImageViewF);
    size += node.NumPorts() * sizeof(MutableImageViewF);
    size = RoundUp(size, 8);  // func_arg_offset_ is aligned.
    size += node.ArgSize();
    PIK_CHECK(size % 8 == 0);
    return size;
  }

  PIK_INLINE SinkTLS* Next() { return end_; }
  PIK_INLINE const SinkTLS* Next() const { return end_; }

  PIK_INLINE void Run(const RunArg& arg) const {
    const int32_t x = x_from_ix(arg.tile_ix);
    const int32_t y = y_from_iy(arg.tile_iy);

    const uint8_t* func_arg =
        reinterpret_cast<const uint8_t*>(this) + func_arg_offset_;
    const ConstImageViewF* inputs =
        reinterpret_cast<const ConstImageViewF*>(this + 1);
    const MutableImageViewF* PIK_RESTRICT outputs =
        reinterpret_cast<const MutableImageViewF*>(inputs + num_inputs_);

    MutableImageViewF shifted[kMaxPorts];
    for (size_t i = 0; i < num_sinks_; ++i) {
      // Mul is actually faster than shifting by log2(size).
      uint8_t* top_left =
          reinterpret_cast<uint8_t*>(outputs[i].Row(y)) + x * type_size_;
      shifted[i].Init(top_left, outputs[i].bytes_per_row());
    }

    const OutputRegion output_region = {
        x, y,
        // Must use partial size for sink.
        out_xsizes_.Get(arg.is_partial_x), out_ysizes_.Get(arg.is_partial_y),
        out_xsizes_.Get(arg.is_partial_x), out_ysizes_.Get(arg.is_partial_y)};
#if VERBOSE & VERBOSE_TILES
    printf("sink   %5d %5d size %3d x %3d (partial %3d x %3d) in %p out %p\n",
           x, y, output_region.xsize, output_region.ysize,
           output_region.partial_xsize, output_region.partial_ysize,
           inputs[0].ConstRow(0), outputs[0].ConstRow(0));
#endif
    func_(func_arg, inputs, output_region, shifted);
  }

 private:
  uint32_t num_inputs_;  // (Never matches Sentinels::kMagic)
  uint32_t func_arg_offset_;
  uint32_t num_sinks_;
  uint32_t type_size_;
  SinkTLS* end_;

  TFFunc func_;
  FullAndPartial out_xsizes_;
  FullAndPartial out_ysizes_;
  CoordFromIndex x_from_ix;
  CoordFromIndex y_from_iy;
  // Followed by: inputs, outputs, func_arg <- end
};

// Partitioning nodes into sources/nodes/sinks avoids branches during RunGraph.
// Injecting unique `sentinel' values between the partitions is more convenient
// than setting an IsLast flag (difficult given the Foreach interface).
class Sentinels {
 public:
  static uint8_t* Insert(uint8_t* p) {
    const uint64_t actual = kMagic;  // Cannot take address directly.
    memcpy(p, &actual, sizeof(actual));
    return p + sizeof(kMagic);
  }

  // Returns whether p points to a sentinel.
  template <class T>
  static bool Check(const T* p) {
    uint64_t actual;
    memcpy(&actual, p, sizeof(actual));
    return actual == kMagic;
  }

  // Skips past the sentinel value (call after Check returned true).
  template <class T>
  static const uint8_t* Skip(const T* p) {
    return reinterpret_cast<const uint8_t*>(p) + sizeof(kMagic);
  }

  // One value per TLS type.
  static uint64_t TotalSize() { return 3 * sizeof(kMagic); }

 private:
  static constexpr uint64_t kMagic = 0xFEFDFCFBFAF9F8F7ull;
};

// Calls Run() for all *TLS instances in "storage". Returns position in storage
// after the next sentinel.
template <class TLS>
PIK_INLINE const uint8_t* RunTLS(const uint8_t* storage, const RunArg& arg) {
  const TLS* tls = reinterpret_cast<const TLS*>(storage);
  while (!Sentinels::Check(tls)) {
    tls->Run(arg);
    tls = tls->Next();
  }
  return Sentinels::Skip(tls);
}

// Runs the graph to produce output for one tile. "p" is per-thread storage.
PIK_INLINE void RunGraph(const uint8_t* PIK_RESTRICT p, const RunArg& arg) {
  p = RunTLS<SourceTLS>(p, arg);
  p = RunTLS<NodeTLS>(p, arg);
  p = RunTLS<SinkTLS>(p, arg);
}

}  // namespace

std::string Borders::ToString() const {
  char buf[100];
  std::snprintf(buf, sizeof(buf), "%1d< %1d> %1d^ %1dv", borders_[L],
                borders_[R], borders_[T], borders_[B]);
  return buf;
}

TFPortBits AllPorts(const TFNode* node) { return (1u << node->NumPorts()) - 1; }
TFType OutType(const TFNode* node) { return node->OutType(); }

class TFBuilderImpl {
 public:
  // Reserve space to avoid reallocation/invalidating TFNode*.
  TFBuilderImpl() { nodes_.reserve(kMaxNodes); }

  TFNode* AddSource(const char* name, const size_t num_ports,
                    const TFType out_type, const TFWrap wrap,
                    const Scale& out_scale) {
    PIK_CHECK(!IsFinalized());

    // Partitioning requires sources to be added before any other node.
    PIK_CHECK(num_nodes_ == 0);
    num_sources_ += 1;

    const uint32_t index = nodes_.size();
    PIK_CHECK(index < kMaxNodes);
    nodes_.emplace_back(index, name, num_ports, out_type, wrap, out_scale);
    return &nodes_.back();
  }

  TFNode* Add(const char* name, const Borders& in_borders,
              const Scale& out_scale, std::vector<TFPorts>&& inputs,
              const size_t num_ports, const TFType out_type, const TFFunc func,
              const uint8_t* arg, const size_t arg_size) {
    PIK_CHECK(!IsFinalized());

    num_nodes_ += 1;

    const uint32_t index = nodes_.size();
    PIK_CHECK(index < kMaxNodes);
    nodes_.emplace_back(index, name, in_borders, out_scale, std::move(inputs),
                        num_ports, out_type, func, arg, arg_size);
    return &nodes_.back();
  }

  // "source" actually points to an image of "type".
  void BindSource(TFNode* node, const TFPortIndex idx_port,
                  const ImageF* source, const TFType type) {
    PIK_CHECK(!IsFinalized());

    node->BindSource(idx_port, source, type);
  }

  // "sink" actually points to an image of "type".
  void BindSink(TFNode* node, const TFPortIndex idx_port, const ImageF* sink,
                const TFType type) {
    PIK_CHECK(!IsFinalized());

    node->BindSink(idx_port, sink, type);
    // We defer updating num_sinks_ until Finalize because BindSink may be
    // called multiple times for a particular node (once per port).
  }

  // Freezes all nodes, which prevents any subsequent calls to Add*/Set*.
  TFGraphPtr Finalize(const ImageSize sink_size, const ImageSize tile_size,
                      ThreadPool* pool) {
    const auto x_from_ix = CoordFromIndex::FromTileSize(tile_size.xsize);
    const auto y_from_iy = CoordFromIndex::FromTileSize(tile_size.ysize);

    // Now that all BindSink are done, update num_sinks_ - but only once for
    // this graph, which can no longer change after this.
    if (++num_finalize_ == 1) {
      for (const TFNode& node : nodes_) {
        num_sinks_ += node.IsSink();
      }
      num_nodes_ -= num_sinks_;
      PIK_CHECK(num_sources_ + num_nodes_ + num_sinks_ == nodes_.size());
    }

    // Recursively compute node sizes from back to front.
    for (auto it = nodes_.rbegin(); it != nodes_.rend(); ++it) {
      it->InitR(num_finalize_, &*it, Borders(), x_from_ix, y_from_iy,
                tile_size.xsize, tile_size.ysize, sink_size.xsize,
                sink_size.ysize);
    }

    // Forward pass: decide buffering mode; must call before RecomputeSizes.
    for (TFNode& node : nodes_) {
      node.Finalize();

#if VERBOSE & VERBOSE_NODES
      printf("  %s\n", node.ToString().c_str());
#endif
    }

    RecomputeSizes();
#if VERBOSE & VERBOSE_GRAPH
    printf("Sources: %u; Nodes: %u; Sinks: %u; Buffer: %zu; Storage: %zu\n",
           num_sources_, num_nodes_, num_sinks_, buffers_size_,
           total_size_ - buffers_size_);
#endif

    // Calls CreateInstance.
    return TFGraphPtr(new TFGraph(sink_size, tile_size, pool, this));
  }

  // Allocates memory; caller must DestroyInstance afterwards.
  uint8_t* CreateInstance(const int thread) const {
    PIK_CHECK(IsFinalized());

    // Must not be called concurrently for the same graph (see TFNode::Port).
    PIK_CHECK(!create_instance_busy_.test_and_set(std::memory_order_acquire));

    for (const TFNode& node : nodes_) {
      node.ResetTLS();
    }

    // Per-thread offset reduces the likelihood of 2K aliasing.
    const size_t offset = (thread % 8) * kImageAlign * 3;
    uint8_t* const allocated =
        static_cast<uint8_t*>(CacheAligned::Allocate(total_size_, offset));

    // Callers assume we return (node) storage, so that comes first. This also
    // ensures convolve.h can load the previous vector from any buffer.
    uint8_t* PIK_RESTRICT storage = allocated;
    const size_t storage_size = total_size_ - buffers_size_;
    uint8_t* PIK_RESTRICT buffers = allocated + storage_size;
    // *Start* of buffer is kImageAlign but rows are only kMaxVectorSize.
    PIK_CHECK(reinterpret_cast<uintptr_t>(buffers) % kImageAlign == 0);

    ForeachSource([&storage, &buffers](const TFNode* node) {
      SourceTLS* tls = reinterpret_cast<SourceTLS*>(storage);
      buffers = tls->Init(*node, buffers);
      storage = reinterpret_cast<uint8_t*>(tls->Next());
    });
    storage = Sentinels::Insert(storage);

    ForeachNode([&storage, &buffers](const TFNode* node) {
      NodeTLS* tls = reinterpret_cast<NodeTLS*>(storage);
      buffers = tls->Init(*node, buffers);
      storage = reinterpret_cast<uint8_t*>(tls->Next());
    });
    storage = Sentinels::Insert(storage);

    ForeachSink([&storage](const TFNode* node) {
      SinkTLS* tls = reinterpret_cast<SinkTLS*>(storage);
      tls->Init(*node);
      storage = reinterpret_cast<uint8_t*>(tls->Next());
    });
    storage = Sentinels::Insert(storage);

    // Skip padding so that the comparison below succeeds.
    const size_t remainder = reinterpret_cast<uintptr_t>(storage) % kImageAlign;
    if (remainder != 0) {
      storage += kImageAlign - remainder;
    }

    PIK_CHECK(storage == allocated + storage_size);
    PIK_CHECK(buffers == allocated + total_size_);

    create_instance_busy_.clear(std::memory_order_release);
    return allocated;
  }

  static void DestroyInstance(uint8_t* tls) {
    // POD, no need to call dtors.
    CacheAligned::Free(tls);
  }

 private:
  bool IsFinalized() const { return num_finalize_ != 0; }

  template <class Visitor>
  void ForeachSource(const Visitor& visitor) const {
    for (size_t i = 0; i < num_sources_; ++i) {
      const TFNode* node = &nodes_[i];
      PIK_CHECK(node->IsSource());
      visitor(node);
    }
  }

  template <class Visitor>
  void ForeachNode(const Visitor& visitor) const {
    for (size_t i = num_sources_; i < num_sources_ + num_nodes_; ++i) {
      const TFNode* node = &nodes_[i];
      PIK_CHECK(!node->IsSource() && !node->IsSink());
      visitor(node);
    }
  }

  template <class Visitor>
  void ForeachSink(const Visitor& visitor) const {
    for (size_t i = 0; i < num_sinks_; ++i) {
      const TFNode* node = &nodes_[num_sources_ + num_nodes_ + i];
      PIK_CHECK(node->IsSink());
      visitor(node);
    }
  }

  void RecomputeSizes() {
    total_size_ = Sentinels::TotalSize();
    buffers_size_ = 0;

    size_t* total = &total_size_;
    size_t* buffers = &buffers_size_;
    ForeachSource([total, buffers](const TFNode* node) {
      *total += SourceTLS::Size(*node);
      *buffers += node->TotalBufferSize();
    });
    ForeachNode([total, buffers](const TFNode* node) {
      *total += NodeTLS::Size(*node);
      *buffers += node->TotalBufferSize();
    });
    ForeachSink([total, buffers](const TFNode* node) {
      *total += SinkTLS::Size(*node);
      *buffers += node->TotalBufferSize();
    });

    // Alignment for first buffer.
    total_size_ = (total_size_ + kImageAlign - 1) & ~(kImageAlign - 1);

    total_size_ += buffers_size_;
  }

  // For partitioning into source/node/sink and verifying their order.
  uint32_t num_sources_ = 0;
  uint32_t num_nodes_ = 0;
  uint32_t num_sinks_ = 0;

  // Only allow Add/Set when zero; otherwise skip parts of subsequent Finalize.
  uint32_t num_finalize_ = 0;

  // Dynamic allocation due to size. Ctor calls reserve to prevent resizing.
  std::vector<TFNode> nodes_;

  // How many bytes each call to CreateInstance (one per thread) allocates.
  size_t total_size_ = 0;
  // How many of those bytes are for node buffers.
  size_t buffers_size_ = 0;

  mutable std::atomic_flag create_instance_busy_ = ATOMIC_FLAG_INIT;
};

TFGraph::TFGraph(const ImageSize sink_size, const ImageSize tile_size,
                 ThreadPool* pool, const TFBuilderImpl* builder)
    : num_tiles_x_(CeilDiv(sink_size.xsize, tile_size.xsize)),
      num_tiles_y_(CeilDiv(sink_size.ysize, tile_size.ysize)),
      num_tiles_(num_tiles_x_ * num_tiles_y_),
      pool_(pool),
      num_instances_(std::max<size_t>(pool->NumThreads(), 1)) {
  for (int i = 0; i < num_instances_; ++i) {
    instances_[i] = builder->CreateInstance(i);
  }
#if VERBOSE & VERBOSE_NODES
  printf("sink_size %d x %d tile_size %d x %d; instances %d\n", sink_size.xsize,
         sink_size.ysize, tile_size.xsize, tile_size.ysize, num_instances_);
#endif
}

TFGraph::~TFGraph() {
  for (int i = 0; i < num_instances_; ++i) {
    TFBuilderImpl::DestroyInstance(instances_[i]);
  }
}

void TFGraph::Run() {
  const TFGraph* self = this;  // For lambda captures.
  const int mask = num_tiles_x_ - 1;

  // Preferred: enough strips to keep threads busy (better locality).
  if (num_tiles_y_ >= pool_->NumThreads() * 2) {
    pool_->Run(0, num_tiles_y_, [self](const int task, const int thread) {
      const TileIndex tile_iy = task;
      RunArg arg(0, tile_iy, self->num_tiles_x_, self->num_tiles_y_);
      for (TileIndex tile_ix = 0; tile_ix < self->num_tiles_x_ - 1; ++tile_ix) {
        arg.tile_ix = tile_ix;
        RunGraph(self->instances_[thread], arg);
      }

      arg.tile_ix = self->num_tiles_x_ - 1;
      arg.is_partial_x = 1;
      RunGraph(self->instances_[thread], arg);
    });
    return;
  }

  // Second-best: scatter tiles but avoid Divider.
  if ((num_tiles_x_ & mask) == 0) {  // Power of two (including 1)
    const int shift = FloorLog2Nonzero(num_tiles_x_);
    pool_->Run(0, num_tiles_,
               [mask, shift, self](const int task, const int thread) {
                 const TileIndex tile_ix = task & mask;
                 const TileIndex tile_iy = task >> shift;
                 const RunArg arg(tile_ix, tile_iy, self->num_tiles_x_,
                                  self->num_tiles_y_);
                 RunGraph(self->instances_[thread], arg);
               });
    return;
  }

  // Fallback: expand task into x,y via 'division'.
  const Divider divide(num_tiles_x_);
  pool_->Run(0, num_tiles_, [&divide, self](const int task, const int thread) {
    const TileIndex tile_iy = divide(task);
    // Remainder - subtracting after mul is faster than modulo.
    const TileIndex tile_ix = task - tile_iy * self->num_tiles_x_;
    const RunArg arg(tile_ix, tile_iy, self->num_tiles_x_, self->num_tiles_y_);
    RunGraph(self->instances_[thread], arg);
  });
}

TFBuilder::TFBuilder() : impl_(new TFBuilderImpl) {}
TFBuilder::~TFBuilder() {}

TFNode* TFBuilder::AddSource(const char* name, const size_t num_ports,
                             const TFType out_type, const TFWrap wrap,
                             const Scale& out_scale) {
  return impl_->AddSource(name, num_ports, out_type, wrap, out_scale);
}

TFNode* TFBuilder::Add(const char* name, const Borders& in_borders,
                       const Scale& out_scale, std::vector<TFPorts>&& inputs,
                       const size_t num_ports, const TFType out_type,
                       const TFFunc func, const uint8_t* arg,
                       const size_t arg_size) {
  return impl_->Add(name, in_borders, out_scale, std::move(inputs), num_ports,
                    out_type, func, arg, arg_size);
}

void TFBuilder::BindSource(TFNode* node, TFPortIndex idx_port,
                           const ImageF* source, TFType type) {
  impl_->BindSource(node, idx_port, source, type);
}

void TFBuilder::BindSink(TFNode* node, TFPortIndex idx_port, const ImageF* sink,
                         TFType type) {
  impl_->BindSink(node, idx_port, sink, type);
}

TFGraphPtr TFBuilder::Finalize(const ImageSize sink_size,
                               const ImageSize tile_size, ThreadPool* pool) {
  return impl_->Finalize(sink_size, tile_size, pool);
}

}  // namespace pik
