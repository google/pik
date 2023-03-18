> **Note**
> There is currently no development going on here. Parts of this project have been used in [JPEG XL](https://github.com/libjxl/libjxl), which is under active development.


# PIK

[![Build status][build-status-img]][build-status]

PIK is a well-rounded image format for photos and the internet.

### Project Goals

PIK is a modernized variant of JPEG with similar goals: efficient storage and
delivery of photos and web images. It is designed from the ground up for
**high quality** and **fast decoding**.

Features enabling high quality (perceptually lossless):

*   built-in support for psychovisual modeling via adaptive quantization and
    XYB color space
*   powerful coding tools: 4x4..32x32 DCT, AC/DC predictors, chroma from luma,
    nonlinear loop filter, enhanced DC precision
*   full-precision (32-bit float) processing, plus support for wide gamut and
    high dynamic range

In addition to fully- and perceptually lossless encodings, PIK achieves a good
balance of quality/size/speed across a wide range of bitrates (0.5 - 3 bpp).
PIK enables automated/unsupervised compression because it guarantees that the
target quality is maintained over the entire image. It prioritizes
**authenticity**, a faithful representation of the original, over aesthetics
achievable by by hallucinating details or 'enhancing' (e.g.
sharpening/saturating) the input.

Features enabling fast decoding (> 1 GB/s multithreaded):

*   Parallel processing of large images
*   SIMD/GPU-friendly, benefits from SSE4 or AVX2
*   Cache-friendly layout
*   Fast and effective entropy coding: context modeling with clustering, rANS

Other features:

*   Alpha, animations, color management, 8-32 bits per channel
*   Flexible progressive scans (quality/resolution/viewport)
*   Graceful upgrade path for existing JPEGs: can reduce their size by 22%
    without loss; or low generation loss because both use 8x8 DCT
*   New, compact and extensible container format
*   Royalty-free commitment

PIK's responsive mode encoder supports passes equivalent to lowering the
resolution by 4x or 8x. The format supports more flexible passes, with any level
of detail from equivalent to 8x downsampling to full resolution.  The amount of
detail in a pass does not need to be uniform: areas of the image can be sent
with higher detail. The impact of responsive mode on encoded image size is low,
averaging to about 2% for a 3 pass responsive image (8x, 4x, full resolution).
In such a configuration, first two passes take on average 20% of image size each.

### Size comparisons

*   PIK achieves perceptually lossless encodings at about 40% of the JPEG
    bitrate.
*   PIK can also store fully lossless at about 75% of 8-bit PNG size (or even
    60% for 16-bit).
*   PIK's perceptually lossless encodings are about 10% of the size of a fully
    lossless representation.

### Build instructions

The software currently requires an AVX2 and FMA capable CPU, e.g. Haswell.
Building currently requires clang 6 or newer.

In order to build, the following instructions can be used:

```console
git submodule update --init
mkdir build
cd build
cmake -DCMAKE_BUILD_TYPE=Release ..
make -j8
```

This creates `cpik` and `dpik` binaries in `build/`.

### Usage

Basic usage is as follows:

```console
cpik [--distance <d>] input.png output.pik
```

The optional `--distance` command line argument to cpik is a Butteraugli
distance (see http://github.com/google/butteraugli), which indicates the largest
acceptable error. Larger values lead to smaller files and lower quality. The
default value of 1.0 should yield a perceptually lossless result.

Note that the bitstream is still under development and not yet frozen.

### Example Usage

Encoding an image to pik ```cpik in_rgb_8bit.png maxError[0.5 .. 3.0] out.pik```

Decoding a pik to an image ```dpik in.pik out_rgb_8bit.png```

### Related projects

*   [JPEG XL](https://github.com/libjxl/libjxl) (reference implementation of image format)
*   [Butteraugli](http://github.com/google/butteraugli) (HVS-aware image differences)
*   [Guetzli](http://github.com/google/guetzli) (JPEG encoder with denser packing)

[build-status]:     https://travis-ci.org/google/pik
[build-status-img]: https://travis-ci.org/google/pik.svg?branch=master
