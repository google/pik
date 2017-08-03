// Copyright 2017 Google Inc. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#define _CRT_SECURE_NO_WARNINGS

#include "image_io.h"
#include "pik.h"

#include <stdio.h>
#include <stdlib.h>

namespace pik {
namespace {

// main() function, within namespace for convenience.
int Compress(const char* pathname_in, const char* distance,
             const char* pathname_out) {
  return 0;
}

}  // namespace
}  // namespace pik

void Usage() {
	fprintf(stderr,
		"PIK Image compressor. Usage: \n"
		"cpik [flags] input_filename output_filename\n"
		"\n"
		"Flags:\n"
		"  --verbose          - Print a verbose trace of all attempts to output.\n"
		"\n"
		"  Only one of the following flags you can set:\n"
		"\n"
		"  --quality Q        - quality, as a butteraugli range distance. [0.5 to 3.0]\n"
		"  --target_bitrate B - Taget bitrate.\n"
		"  --fast_mode        - If set, will use a compression method that is fast.\n"
		"  --uniform_quant U  - 0.1 or above mean quantize everywhere with that value.\n");
	exit(1);
}

int main(int argc, char** argv) {
  pik::Image3B in;
  pik::CompressParams params;
  int verbose = 0;
  int only = 0;
  float butteraugli_distance = NULL;
  float target_bitrate = NULL;
  int fast_mode = 0;
  float uniform_quant = NULL;

  int arg_idx = 1;
  for(; arg_idx < argc; arg_idx++) {
    if (strnlen(argv[arg_idx], 2) < 2 || argv[arg_idx][0] != '-' || argv[arg_idx][1] != '-')
      break;
    if (!strcmp(argv[arg_idx], "--verbose")) {
      verbose = 1;
    } else if (!strcmp(argv[arg_idx], "--distance")|| !strcmp(argv[arg_idx], "--quality")) {
      arg_idx++;
      if (arg_idx >= argc || only)
        Usage();
	  butteraugli_distance = strtod(argv[arg_idx], nullptr);
	  params.butteraugli_distance = butteraugli_distance;
	  only = 1;
    } else if (!strcmp(argv[arg_idx], "--target_bitrate")) {
      arg_idx++;
      if (arg_idx >= argc || only)
        Usage();
	  target_bitrate = strtod(argv[arg_idx], nullptr);
	  params.target_bitrate = target_bitrate;
	  only = 1;
	} else if (!strcmp(argv[arg_idx], "--fast_mode")) {
      arg_idx++;
      if (arg_idx >= argc || only)
        Usage();
	  fast_mode = 1;
	  params.fast_mode = fast_mode;
	  only = 1;
	} else if (!strcmp(argv[arg_idx], "--uniform_quant")) {
      arg_idx++;
      if (arg_idx >= argc || only)
        Usage();
	  uniform_quant = strtod(argv[arg_idx], nullptr);
	  params.uniform_quant = uniform_quant;
	  only = 1;
	} else if (!strcmp(argv[arg_idx], "--")) {
      arg_idx++;
      break;
    } else {
      fprintf(stderr, "Unknown commandline flag: %s\n", argv[arg_idx]);
      Usage();
    }
  }

  if (argc - arg_idx != 2) {
    Usage();
  }

  const char* pathname_in = argv[arg_idx];
  const char* pathname_out = argv[arg_idx + 1];

  if (!ReadImage(pik::ImageFormatPNG(), pathname_in, &in)) {
    fprintf(stderr, "Failed to open %s.\n", pathname_in);
    return 1;
  }

  if (butteraugli_distance != NULL && !(0.5f <= butteraugli_distance && butteraugli_distance <= 3.0f)) {
    fprintf(stderr, "Invalid/out of range distance '%s', try 0.5 to 3.\n",
            butteraugli_distance);
    return 1;
  }

  if(butteraugli_distance >= 0.5) {
    printf("Compressing with maximum Butteraugli distance %f\n",
           butteraugli_distance);
  }

  if (target_bitrate > 0.0) {
	  printf("Compressing with target bitrate value %f\n",
		  target_bitrate);
  }

  if (fast_mode) {
	  printf("Compressing with fast mode on\n");
  }

  if (uniform_quant > 0.0) {
	  printf("Compressing with uniform quant value %f\n",
		  uniform_quant);
  }

  pik::Bytes compressed;
  pik::PikInfo aux_out;
  if (PixelsToPik(params, in, &compressed, &aux_out) != pik::Status::OK) {
    fprintf(stderr, "Failed to compress.\n");
    return 1;
  }

  printf("Compressed to %zu bytes\n", compressed.size());

  FILE* f = fopen(pathname_out, "wb");
  if (f == nullptr) {
    fprintf(stderr, "Failed to open %s.\n", pathname_out);
    return 1;
  }
  const size_t bytes_written =
      fwrite(compressed.data(), 1, compressed.size(), f);
  if (bytes_written != compressed.size()) {
    fprintf(stderr, "I/O error, only wrote %zu bytes.\n", bytes_written);
    return 1;
  }
  fclose(f);
  return 0;
}

