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

#include "jpeg_data_decoder.h"

#include <setjmp.h>
#include <cstdint>
#include <memory>
#include <string>

extern "C" {
#include "jconfig.h"
#include "jmorecfg.h"
#include "jpeglib.h"
}
extern "C" {
#include "jpegint.h"  // NOLINT (order dependency)
}

#include "guetzli/jpeg_error.h"

// Handler for fatal JPEG library errors: clean up & return
// Same as in //util/jpeg/jpeg_mem.cc
METHODDEF(void)jpeg_catch_error(j_common_ptr cinfo) {
  (*cinfo->err->output_message) (cinfo);
  jmp_buf* jpeg_jmpbuf = reinterpret_cast<jmp_buf*>(cinfo->client_data);
  jpeg_destroy(cinfo);
  longjmp(*jpeg_jmpbuf, 1);
}

// Define a dummy start_input_pass(), since we don't have to any anything with
// the input stream.
METHODDEF(void)dummy_start_input_pass(j_decompress_ptr cinfo) {
}

namespace pik {

static const int kThumbRatio = 8;
static const int ApplyRatio(int value, int ratio) {
  return (value + ratio - 1) / ratio;
}

// Mimic libjpeg's heuristics to guess jpeg color space.
J_COLOR_SPACE GuessJpegColorSpace(const guetzli::JPEGData& jpg) {
  bool has_JFIF_marker = false;
  bool has_Adobe_marker = false;
  uint8_t Adobe_transform = 0;
  for (const std::string& app : jpg.app_data) {
    if (app[0] == 0xe0) {
      has_JFIF_marker = true;
    } else if (app[0] == 0xee && app.size() >= 15) {
      has_Adobe_marker = true;
      Adobe_transform = app[14];
    }
  }
  if (jpg.components.size() == 1) {
    return JCS_GRAYSCALE;
  } else if (jpg.components.size() == 3) {
    if (has_JFIF_marker) {
      return JCS_YCbCr;
    } else if (has_Adobe_marker) {
      return (Adobe_transform == 0) ? JCS_RGB : JCS_YCbCr;
    } else {
      const int cid0 = jpg.components[0].id;
      const int cid1 = jpg.components[1].id;
      const int cid2 = jpg.components[2].id;
      if (cid0 == 'R' && cid1 == 'G' && cid2 == 'B') {
        return JCS_RGB;
      } else {
        return JCS_YCbCr;
      }
    }
  }
  return JCS_UNKNOWN;
}

// Sets all fields of cinfo that would otherwise be set by jpeg_read_header().
void SetHeaderInfoFromJPEGData(const guetzli::JPEGData& jpg,
                               j_decompress_ptr cinfo,
                               bool for_thumbnail) {
  cinfo->image_width = jpg.width;
  cinfo->image_height = jpg.height;
  cinfo->num_components = jpg.components.size();
  cinfo->data_precision = 8;
  cinfo->progressive_mode = FALSE;
  cinfo->arith_code = FALSE;
  cinfo->max_h_samp_factor = jpg.max_h_samp_factor;
  cinfo->max_v_samp_factor = jpg.max_v_samp_factor;
  cinfo->total_iMCU_rows = jpg.MCU_rows;
  cinfo->jpeg_color_space = GuessJpegColorSpace(jpg);
  cinfo->out_color_space = JCS_RGB;
  cinfo->scale_num = 1;
  cinfo->scale_denom = for_thumbnail ? kThumbRatio : 1;
  cinfo->output_gamma = 1.0;
  cinfo->buffered_image = TRUE;
  cinfo->raw_data_out = FALSE;
  cinfo->dct_method = JDCT_DEFAULT;
  cinfo->do_fancy_upsampling = TRUE;
  cinfo->do_block_smoothing = for_thumbnail ? FALSE : TRUE;
  cinfo->quantize_colors = FALSE;
  cinfo->input_scan_number = 1;
  cinfo->input_iMCU_row = cinfo->total_iMCU_rows;
  cinfo->inputctl->eoi_reached = TRUE;
  cinfo->inputctl->start_input_pass = dummy_start_input_pass;
  cinfo->global_state = DSTATE_READY;
  cinfo->comp_info = reinterpret_cast<jpeg_component_info *>(
      cinfo->mem->alloc_small(
          reinterpret_cast<j_common_ptr>(cinfo), JPOOL_IMAGE,
          cinfo->num_components * sizeof(jpeg_component_info)));
  for (int i = 0; i < jpg.components.size(); ++i) {
    const guetzli::JPEGComponent& c = jpg.components[i];
    const guetzli::JPEGQuantTable& q = jpg.quant[c.quant_idx];
    jpeg_component_info * compptr = &cinfo->comp_info[i];
    compptr->component_index = i;
    compptr->component_id = c.id;
    compptr->h_samp_factor = c.h_samp_factor;
    compptr->v_samp_factor = c.v_samp_factor;
    compptr->quant_tbl_no = c.quant_idx;
    compptr->width_in_blocks = c.width_in_blocks;
    compptr->height_in_blocks = c.height_in_blocks;
    compptr->component_needed = TRUE;
    compptr->quant_table = jpeg_alloc_quant_table((j_common_ptr) cinfo);
    for (int k = 0; k < guetzli::kDCTBlockSize; ++k) {
      compptr->quant_table->quantval[k] = (UINT16)q.values[k];
    }
    // Note that the LIBJPEG_TURBO_VERSION_NUMBER macro was introduced between
    // 1.4.90 (1.5beta1) and 1.5.0. As a result, if brunsli is built using the
    // libjpeg-turbo 1.5beta1 release, the following fields will be left
    // uninitialized, leading to incorrect decodes.
#if LIBJPEG_TURBO_VERSION_NUMBER >= 1004090
    cinfo->master->first_MCU_col[i] = 0;
    cinfo->master->last_MCU_col[i] = compptr->width_in_blocks - 1;
#endif
  }
}

void SetCoeffsFromJPEGData(const guetzli::JPEGData& jpg, j_decompress_ptr cinfo,
                           bool for_thumbnail) {
  for (int ci = 0; ci < jpg.components.size(); ci++) {
    const guetzli::JPEGComponent& c = jpg.components[ci];
    const guetzli::coeff_t* coeffs = &c.coeffs[0];
    for (int y = 0; y < c.height_in_blocks; ++y) {
      JBLOCKARRAY row_ptr = (*cinfo->mem->access_virt_barray)
          ((j_common_ptr)cinfo, cinfo->coef->coef_arrays[ci],
           y, (JDIMENSION)1, TRUE);
      for (int x = 0; x < c.width_in_blocks; ++x) {
        if (for_thumbnail) {
          row_ptr[0][x][0] = (JCOEF)coeffs[0];
          for (int k = 1; k <DCTSIZE2; ++k) {
            row_ptr[0][x][k] = 0;
          }
        } else {
          for (int k = 0; k < DCTSIZE2; ++k) {
            row_ptr[0][x][k] = (JCOEF)coeffs[k];
          }
        }
        coeffs += DCTSIZE2;
      }
    }
  }
}

bool DecodeJpegToRGB(const guetzli::JPEGData& jpg, uint8_t* rgb,
                     bool thumbnail) {
  int ok = true;

  if (jpg.components.size() != 1 && jpg.components.size() != 3) {
    return false;
  }

  // Initialize libjpeg structures. Modify the usual jpeg error manager to
  // catch fatal errors (as in //util/jpeg/jpeg_mem.cc).
  jpeg_decompress_struct cinfo;
  jpeg_error_mgr jerr;
  cinfo.err = jpeg_std_error(&jerr);
  jmp_buf jpeg_jmpbuf;
  cinfo.client_data = &jpeg_jmpbuf;
  jerr.error_exit = jpeg_catch_error;
  if (setjmp(jpeg_jmpbuf)) {
    return false;
  }
  jpeg_create_decompress(&cinfo);

  // Instead of calling jpeg_read_header(), set the relevant fields of cinfo
  // using the data in jpg.
  SetHeaderInfoFromJPEGData(jpg, &cinfo, thumbnail);

  jpeg_start_decompress(&cinfo);

  SetCoeffsFromJPEGData(jpg, &cinfo, thumbnail);

  jpeg_start_output(&cinfo, 1);

  JSAMPLE* output_line = static_cast<JSAMPLE*>(rgb);
  const int stride = 3 * cinfo.output_width;
  while (cinfo.output_scanline < cinfo.output_height) {
    int num_lines_read = jpeg_read_scanlines(&cinfo, &output_line, 1);
    if (num_lines_read == 1) {
      output_line += stride;
    } else {
      ok = false;
      cinfo.output_scanline = cinfo.output_height;
    }
  }

  jpeg_finish_output(&cinfo);
  jpeg_destroy_decompress(&cinfo);

  return ok;
}

std::vector<uint8_t> DecodeJpegToRGB(const guetzli::JPEGData& jpg,
                                     bool thumbnail) {
  const int ratio = thumbnail ? 8 : 1;
  const int width = ApplyRatio(jpg.width, ratio);
  const int height = ApplyRatio(jpg.height, ratio);
  std::vector<uint8_t> data(3 * width * height);
  if (!DecodeJpegToRGB(jpg, &data[0], thumbnail)) {
    data.resize(0);
  }
  return data;
}

}  // namespace pik
