Pik is a new lossy image format for the internet. This directory contains
an encoder and a decoder for the format.

This project is in the initial research stage, please don't use it for any
purpose.

The software currently requires an AVX2 and FMA capable CPU, e.g. Haswell.

### Build instructions

Please ensure you have the libpng-dev and libjpeg-dev packages installed.
Then simply run `make -j8`, which creates cpik and dpik binaries in bin/.

The second command line argument to cpik is a Butteraugli distance (see
http://github.com/google/butteraugli), which indicates the largest acceptable
error. Larger values lead to smaller files and lower quality. Try 1.0 for a
visually lossless result.

### Example Usage

Encoding an image to pik ```cpik in_rgb_8bit.png maxError[0.5 .. 3.0] out.pik```

Decoding a pik to an image ```dpik in.pik out_rgb_8bit.png```

### Related projects

*   Butteraugli (HVS-aware image differences)
*   Brunsli (lossless JPEG repacker)
*   Guetzli (JPEG encoder with denser packing)

This is not an official Google product.
