#ifndef TRANSFORM_H_
#define TRANSFORM_H_

#include "data_parallel.h"
#include "image.h"

namespace pik {

class OpsinIntraTransform {
 public:
  virtual ~OpsinIntraTransform() = default;
  // Computes an intra-frame encoding of the given rectangle in img.
  virtual void OpsinToIntraInPlace(Image3F* img, const Rect& rect,
                                   ThreadPool* pool) = 0;

  // Same as OpsinToIntraInPlace, but does not modify img. The returned image is
  // full-size, but the value of pixels outside rect are undefined.
  // The given rectangle in the output image will be valid at least until
  // the next call to this function with an overlapping rect. In particular,
  // multiple call to this method or to IntraToOpsin with the same rect are
  // thread-hostile.
  virtual const Image3F& OpsinToIntra(const Image3F& img, const Rect& rect,
                                      ThreadPool* pool) = 0;

  // Inverse of OpsinToIntraInPlace.
  virtual void IntraToOpsinInPlace(Image3F* img, const Rect& rect,
                                   ThreadPool* pool) = 0;
  // Same as IntraToOpsinInPlace, but do not modify img. The returned image is
  // full-size, but the value of pixels outside rect are undefined.
  // The given rectangle in the output image will be valid at least until
  // the next call to this function with an overlapping rect. In particular,
  // multiple call to this method or to OpsinToIntra with the same rect are
  // thread-hostile.
  virtual const Image3F& IntraToOpsin(const Image3F& img, const Rect& rect,
                                      ThreadPool* pool) = 0;
};

class NoopOpsinIntraTransform : public OpsinIntraTransform {
 public:
  void OpsinToIntraInPlace(Image3F* img, const Rect& rect,
                           ThreadPool* pool) override {}

  const Image3F& OpsinToIntra(const Image3F& img, const Rect& rect,
                              ThreadPool* pool) override {
    return img;
  }

  void IntraToOpsinInPlace(Image3F* img, const Rect& rect,
                           ThreadPool* pool) override {}

  const Image3F& IntraToOpsin(const Image3F& img, const Rect& rect,
                              ThreadPool* pool) override {
    return img;
  }
};

}  // namespace pik

#endif  // TRANSFORM_H_
