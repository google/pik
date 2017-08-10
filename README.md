# Pik

[![Build status][build-status-img]][build-status]

Pik is a new lossy image format for the internet. This directory contains
an encoder and a decoder for the format.

This project is in the initial research stage, please don't use it for any
purpose.

### Project Goals

Pik is to have roughly the same performance requirements space that
JPEG is holding today. We aim to have roughly similar decoding speed,
i.e., no more than 50 % slower in decoding, but adding more modern
compression and prediction methods, giving 55 % more density. We plan to
improve to 65 % before freezing the format. Some of these new advances may
come with a moderate drop in decoding speed (possibly 40 % of jpeg speed).

We are planning for a tiled format so that multi-core decoding can decode
a single image faster (possibly around 16x). The current version has no
support for this yet.

We are planning to keep the format 8x8 DCT based, possibly with some support
for non-integral-transform-based direct mode blocks (or overlay blocks).

### Build instructions

The software currently requires an AVX2 and FMA capable CPU, e.g. Haswell.

Please ensure you have the libpng-dev and libjpeg-dev packages installed.
Then simply run `make -j8`, which creates cpik and dpik binaries in bin/.

The second command line argument to cpik is a Butteraugli distance (see
http://github.com/google/butteraugli), which indicates the largest acceptable
error. Larger values lead to smaller files and lower quality. Try 1.0 for a
visually lossless result.

### Related projects

*   Butteraugli (HVS-aware image differences)
*   Guetzli (JPEG encoder with denser packing)

This is not an official Google product.

[build-status]:     https://travis-ci.org/google/pik
[build-status-img]: https://travis-ci.org/google/pik.svg?branch=master
