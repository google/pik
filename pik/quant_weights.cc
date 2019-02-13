// Copyright 2017 Google Inc. All Rights Reserved.
//
// Use of this source code is governed by an MIT-style
// license that can be found in the LICENSE file or at
// https://opensource.org/licenses/MIT.
#include "pik/quant_weights.h"

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <limits>
#include "pik/bit_reader.h"
#include "pik/common.h"
#include "pik/dct.h"
#include "pik/pik_info.h"
#include "pik/status.h"

namespace pik {

// kQuantWeights[N * N * c + N * y + x] is the relative weight of the (x, y)
// coefficient in component c. Higher weights correspond to finer quantization
// intervals and more bits spent in encoding.

namespace {
void GetQuantWeightsDCT2(const float dct2weights[3][6], double* weights) {
  for (size_t c = 0; c < 3; c++) {
    size_t start = c * 64;
    weights[start] = 0xBAD;
    weights[start + 1] = weights[start + 8] = dct2weights[c][0];
    weights[start + 9] = dct2weights[c][1];
    for (size_t y = 0; y < 2; y++) {
      for (size_t x = 0; x < 2; x++) {
        weights[start + y * 8 + x + 2] = dct2weights[c][2];
        weights[start + (y + 2) * 8 + x] = dct2weights[c][2];
      }
    }
    for (size_t y = 0; y < 2; y++) {
      for (size_t x = 0; x < 2; x++) {
        weights[start + (y + 2) * 8 + x + 2] = dct2weights[c][3];
      }
    }
    for (size_t y = 0; y < 4; y++) {
      for (size_t x = 0; x < 4; x++) {
        weights[start + y * 8 + x + 4] = dct2weights[c][4];
        weights[start + (y + 4) * 8 + x] = dct2weights[c][4];
      }
    }
    for (size_t y = 0; y < 4; y++) {
      for (size_t x = 0; x < 4; x++) {
        weights[start + (y + 4) * 8 + x + 4] = dct2weights[c][5];
      }
    }
  }
}

const double* GetQuantWeightsLines() {
  // The first value does not matter: it is the DC which is quantized elsewhere.
  static const double kPositionWeights[64] = {
      0,   100, 100, 100, 100, 100, 100, 5, 100, 100, 50, 20, 20, 10, 5, 5,
      100, 100, 50,  20,  20,  10,  5,   5, 100, 50,  50, 20, 20, 10, 5, 5,
      100, 20,  20,  20,  20,  10,  5,   5, 100, 10,  10, 10, 10, 10, 5, 5,
      100, 5,   5,   5,   5,   5,   5,   5, 5,   5,   5,  5,  5,  5,  5, 5,
  };
  static const double kChannelWeights[3] = {0.2, 0.5, 0.01};
  static const double kGlobal = 35.0;

  static double kQuantWeights[3 * 8 * 8] = {};

  for (size_t c = 0; c < 3; c++) {
    size_t start = c * 64;
    for (size_t y = 0; y < 8; y++) {
      for (size_t x = 0; x < 8; x++) {
        kQuantWeights[start + y * 8 + x] =
            kPositionWeights[y * 8 + x] * kChannelWeights[c] * kGlobal;
      }
    }
  }
  return kQuantWeights;
}

void GetQuantWeightsIdentity(const float idweights[3][3], double* weights) {
  for (size_t c = 0; c < 3; c++) {
    for (int i = 0; i < 64; i++) {
      weights[64 * c + i] = idweights[c][0];
    }
    weights[64 * c + 1] = idweights[c][1];
    weights[64 * c + 8] = idweights[c][1];
    weights[64 * c + 9] = idweights[c][2];
  }
}

// Computes quant weights for a SX*SY-sized transform, using num_bands
// eccentricity bands and num_ebands eccentricity bands. If print_mode is 1,
// prints the resulting matrix; if print_mode is 2, prints the matrix in a
// format suitable for a 3d plot with gnuplot.
template <size_t SX, size_t SY, size_t print_mode = 0>
Status GetQuantWeights(
    const float distance_bands[3][DctQuantWeightParams::kMaxDistanceBands],
    size_t num_bands,
    const float eccentricity_bands[3][DctQuantWeightParams::kMaxRadialBands],
    size_t num_ebands, double* out) {
  auto mult = [](double v) {
    if (v > 0) return 1 + v;
    return 1 / (1 - v);
  };

  auto interpolate = [](double pos, double max, double* array, size_t len) {
    double scaled_pos = pos * (len - 1) / max;
    size_t idx = scaled_pos;
    PIK_ASSERT(idx + 1 < len);
    double a = array[idx];
    double b = array[idx + 1];
    return a * pow(b / a, scaled_pos - idx);
  };

  for (size_t c = 0; c < 3; c++) {
    if (print_mode) {
      fprintf(stderr, "Channel %lu\n", c);
    }
    double bands[DctQuantWeightParams::kMaxDistanceBands] = {
        distance_bands[c][0]};
    for (size_t i = 1; i < num_bands; i++) {
      bands[i] = bands[i - 1] * mult(distance_bands[c][i]);
      if (bands[i] < 0) return PIK_FAILURE("Invalid distance bands");
    }
    double ebands[DctQuantWeightParams::kMaxRadialBands + 1] = {1.0};
    for (size_t i = 1; i <= num_ebands; i++) {
      ebands[i] = ebands[i - 1] * mult(eccentricity_bands[c][i - 1]);
      if (ebands[i] < 0) return PIK_FAILURE("Invalid eccentricity bands");
    }
    for (size_t y = 0; y < SY; y++) {
      for (size_t x = 0; x < SX; x++) {
        double dx = 1.0 * x / (SX - 1);
        double dy = 1.0 * y / (SY - 1);
        double distance = std::sqrt(dx * dx + dy * dy);
        double wd =
            interpolate(distance, std::sqrt(2) + 1e-6, bands, num_bands);
        double eccentricity =
            (x == 0 && y == 0) ? 0 : std::abs((double)dx - dy) / distance;
        double we =
            interpolate(eccentricity, 1.0 + 1e-6, ebands, num_ebands + 1);
        double weight = we * wd;

        if (print_mode == 1) {
          fprintf(stderr, "%15.12f, ", weight);
        }
        if (print_mode == 2) {
          fprintf(stderr, "%lu %lu %15.12f\n", x, y, weight);
        }
        out[c * SX * SY + y * SX + x] = weight;
      }
      if (print_mode) fprintf(stderr, "\n");
      if (print_mode == 1) fprintf(stderr, "\n");
    }
    if (print_mode) fprintf(stderr, "\n");
  }
  return true;
}

// TODO(veluca): use proper encoding for floats. If not, use integer
// encoding/decoding functions from byte_order.h. Also consider moving those
// fields to use the header machinery.
void EncodeUint(uint32_t v, std::string* s) {
  *s += (uint8_t)(v >> 24);
  *s += (uint8_t)(v >> 16);
  *s += (uint8_t)(v >> 8);
  *s += (uint8_t)v;
}

void EncodeFloat(float v, std::string* s) {
  static_assert(sizeof(float) == sizeof(uint32_t),
                "Float should be composed of 32 bits!");
  uint32_t tmp;
  memcpy(&tmp, &v, sizeof(float));
  EncodeUint(tmp, s);
}

uint32_t DecodeUint(BitReader* br) {
  br->FillBitBuffer();
  uint32_t v = br->ReadBits(8);
  v = (v << 8) | br->ReadBits(8);
  v = (v << 8) | br->ReadBits(8);
  v = (v << 8) | br->ReadBits(8);
  return v;
}

float DecodeFloat(BitReader* br) {
  uint32_t tmp = DecodeUint(br);
  float ret;
  memcpy(&ret, &tmp, sizeof(float));
  return ret;
}

void EncodeDctParams(const DctQuantWeightParams& params, std::string* s) {
  s += (uint8_t)params.num_distance_bands;
  for (size_t c = 0; c < 3; c++) {
    for (size_t i = 0; i < params.num_distance_bands; i++) {
      EncodeFloat(params.distance_bands[c][i], s);
    }
  }
  s += (uint8_t)params.num_eccentricity_bands;
  for (size_t c = 0; c < 3; c++) {
    for (size_t i = 0; i < params.num_eccentricity_bands; i++) {
      EncodeFloat(params.eccentricity_bands[c][i], s);
    }
  }
}

Status DecodeDctParams(BitReader* br, DctQuantWeightParams* params) {
  br->FillBitBuffer();
  if (params->num_distance_bands > DctQuantWeightParams::kMaxDistanceBands)
    return PIK_FAILURE("Too many distance bands");
  if (params->num_distance_bands == 0)
    return PIK_FAILURE("Too few distance bands");
  for (size_t c = 0; c < 3; c++) {
    for (size_t i = 0; i < params->num_distance_bands; i++) {
      params->distance_bands[c][i] = DecodeFloat(br);
    }
  }
  br->FillBitBuffer();
  params->num_eccentricity_bands = br->ReadBits(8);
  if (params->num_eccentricity_bands > DctQuantWeightParams::kMaxRadialBands)
    return PIK_FAILURE("Too many eccentricity bands");
  for (size_t c = 0; c < 3; c++) {
    for (size_t i = 0; i < params->num_eccentricity_bands; i++) {
      params->eccentricity_bands[c][i] = DecodeFloat(br);
    }
  }
  return true;
}

std::string Encode(const QuantEncoding& encoding) {
  std::string out(1, encoding.mode);
  switch (encoding.mode) {
    case QuantEncoding::kQuantModeDefault:
      break;
    case QuantEncoding::kQuantModeID: {
      for (size_t c = 0; c < 3; c++) {
        for (size_t i = 0; i < 3; i++) {
          EncodeFloat(encoding.idweights[c][i], &out);
        }
      }
      break;
    }
    case QuantEncoding::kQuantModeDCT2: {
      for (size_t c = 0; c < 3; c++) {
        for (size_t i = 0; i < 6; i++) {
          EncodeFloat(encoding.dct2weights[c][i], &out);
        }
      }
      break;
    }
    case QuantEncoding::kQuantModeDCT4: {
      for (size_t c = 0; c < 3; c++) {
        for (size_t i = 0; i < 2; i++) {
          EncodeFloat(encoding.dct4multipliers[c][i], &out);
        }
      }
      EncodeDctParams(encoding.dct_params, &out);
      break;
    }
    case QuantEncoding::kQuantModeDCT: {
      EncodeDctParams(encoding.dct_params, &out);
      break;
    }
    case QuantEncoding::kQuantModeRaw: {
      out += (uint8_t)encoding.block_dim;
      for (size_t c = 0; c < 3; c++) {
        for (size_t y = 0; y < encoding.block_dim * kBlockDim; y++) {
          for (size_t x = 0; x < encoding.block_dim * kBlockDim; x++) {
            if (x < encoding.block_dim && y < encoding.block_dim) continue;
            EncodeFloat(
                encoding.weights[c][y * encoding.block_dim * kBlockDim + x],
                &out);
          }
        }
      }
      break;
    }
    case QuantEncoding::kQuantModeRawScaled: {
      out += (uint8_t)encoding.block_dim;
      for (size_t y = 0; y < encoding.block_dim * kBlockDim; y++) {
        for (size_t x = 0; x < encoding.block_dim * kBlockDim; x++) {
          if (x < encoding.block_dim && y < encoding.block_dim) continue;
          EncodeFloat(
              encoding.weights[0][y * encoding.block_dim * kBlockDim + x],
              &out);
        }
      }
      for (size_t c = 0; c < 3; c++) {
        EncodeFloat(encoding.scales[c], &out);
      }
      break;
    }
  }
  return out;
}

Status Decode(BitReader* br, QuantEncoding* encoding, size_t required_size) {
  br->FillBitBuffer();
  int mode = br->ReadBits(8);
  switch (mode) {
    case QuantEncoding::kQuantModeDefault:
      break;
    case QuantEncoding::kQuantModeID: {
      if (required_size != 1) return PIK_FAILURE("Invalid mode");
      for (size_t c = 0; c < 3; c++) {
        for (size_t i = 0; i < 3; i++) {
          encoding->idweights[c][i] = DecodeFloat(br);
        }
      }
      break;
    }
    case QuantEncoding::kQuantModeDCT2: {
      if (required_size != 1) return PIK_FAILURE("Invalid mode");
      for (size_t c = 0; c < 3; c++) {
        for (size_t i = 0; i < 6; i++) {
          encoding->dct2weights[c][i] = DecodeFloat(br);
        }
      }
      break;
    }
    case QuantEncoding::kQuantModeDCT4: {
      if (required_size != 1) return PIK_FAILURE("Invalid mode");
      for (size_t c = 0; c < 3; c++) {
        for (size_t i = 0; i < 2; i++) {
          encoding->dct4multipliers[c][i] = DecodeFloat(br);
        }
      }
      PIK_RETURN_IF_ERROR(DecodeDctParams(br, &encoding->dct_params));
      break;
    }
    case QuantEncoding::kQuantModeDCT: {
      PIK_RETURN_IF_ERROR(DecodeDctParams(br, &encoding->dct_params));
      break;
    }
    case QuantEncoding::kQuantModeRaw: {
      br->FillBitBuffer();
      encoding->block_dim = br->ReadBits(8);
      if (required_size != encoding->block_dim)
        return PIK_FAILURE("Invalid mode");
      for (size_t c = 0; c < 3; c++) {
        for (size_t y = 0; y < encoding->block_dim * kBlockDim; y++) {
          for (size_t x = 0; x < encoding->block_dim * kBlockDim; x++) {
            // Override LLF values in the quantization table with invalid
            // values.
            if (x < encoding->block_dim && y < encoding->block_dim) {
              encoding->weights[c][y * encoding->block_dim * kBlockDim + x] =
                  0xBAD;
              continue;
            }
            encoding->weights[c][y * encoding->block_dim * kBlockDim + x] =
                DecodeFloat(br);
          }
        }
      }
      break;
    }
    case QuantEncoding::kQuantModeRawScaled: {
      br->FillBitBuffer();
      encoding->block_dim = br->ReadBits(8);
      if (required_size != encoding->block_dim)
        return PIK_FAILURE("Invalid mode");
      for (size_t y = 0; y < encoding->block_dim * kBlockDim; y++) {
        for (size_t x = 0; x < encoding->block_dim * kBlockDim; x++) {
          // Override LLF values in the quantization table with invalid values.
          if (x < encoding->block_dim && y < encoding->block_dim) {
            encoding->weights[0][y * encoding->block_dim * kBlockDim + x] =
                0xBAD;
            continue;
          }
          encoding->weights[0][y * encoding->block_dim * kBlockDim + x] =
              DecodeFloat(br);
        }
      }
      for (size_t c = 0; c < 3; c++) {
        encoding->scales[c] = DecodeFloat(br);
      }
      break;
    }
    default:
      return PIK_FAILURE("Invalid quantization table encoding");
  }
  encoding->mode = QuantEncoding::Mode(mode);
  if (!br->Healthy()) return PIK_FAILURE("Failed reading quantization tables");
  return true;
}

Status ComputeQuantTable(const QuantEncoding& encoding, float* table,
                         size_t* offsets, QuantKind kind, size_t* pos) {
  double weights[3 * kMaxQuantTableSize];
  double numerators[kMaxQuantTableSize];
  decltype(&GetQuantWeights<8, 8>) get_dct_weights = nullptr;

  constexpr int N = kBlockDim;
  constexpr int block_size = N * N;
  const float* idct4_scales = IDCTScales<N / 2>();
  const float* idct_scales = IDCTScales<N>();
  const float* idct16_scales = IDCTScales<2 * N>();
  const float* idct32_scales = IDCTScales<4 * N>();
  size_t num = 0;
  switch (kind) {
    case kQuantKindDCT8: {
      num = block_size;
      get_dct_weights = GetQuantWeights<8, 8>;
      for (size_t i = 0; i < num; i++) {
        const size_t x = i % N;
        const size_t y = i / N;
        const float idct_scale = idct_scales[x] * idct_scales[y] / num;
        numerators[i] = idct_scale;
      }
      break;
    }
    case kQuantKindDCT16: {
      num = 4 * block_size;
      get_dct_weights = GetQuantWeights<16, 16>;
      for (size_t i = 0; i < num; i++) {
        const size_t x = i % (2 * N);
        const size_t y = i / (2 * N);
        const float idct_scale = idct16_scales[x] * idct16_scales[y] / num;
        numerators[i] = idct_scale;
      }
      break;
    }
    case kQuantKindDCT32: {
      num = 16 * block_size;
      get_dct_weights = GetQuantWeights<32, 32>;
      for (size_t i = 0; i < num; i++) {
        const size_t x = i % (4 * N);
        const size_t y = i / (4 * N);
        const float idct_scale = idct32_scales[x] * idct32_scales[y] / num;
        numerators[i] = idct_scale;
      }
      break;
    }
    case kQuantKindDCT4: {
      num = block_size;
      get_dct_weights = GetQuantWeights<4, 4>;
      for (size_t i = 0; i < N * N; i++) {
        const size_t x = i % N;
        const size_t y = i / N;
        float idct_scale =
            idct4_scales[x / 2] * idct4_scales[y / 2] / (N / 2 * N / 2);
        numerators[i] = idct_scale;
      }
      break;
    }
    case kQuantKindID:
    case kQuantKindDCT2:
    case kQuantKindLines: {
      get_dct_weights = GetQuantWeights<8, 8>;
      num = block_size;
      std::fill_n(numerators, block_size, 1.0);
      break;
    }
    case kNumQuantKinds: {
      PIK_ASSERT(false);
    }
  }
  PIK_ASSERT(get_dct_weights != nullptr);

  switch (encoding.mode) {
    case QuantEncoding::kQuantModeDefault: {
      // Default quant encoding should get replaced by the actual default
      // parameters by the caller.
      PIK_ASSERT(false);
      break;
    }
    case QuantEncoding::kQuantModeID: {
      PIK_ASSERT(num == block_size);
      GetQuantWeightsIdentity(encoding.idweights, weights);
      break;
    }
    case QuantEncoding::kQuantModeDCT2: {
      PIK_ASSERT(num == block_size);
      GetQuantWeightsDCT2(encoding.dct2weights, weights);
      break;
    }
    case QuantEncoding::kQuantModeDCT4: {
      PIK_ASSERT(num == block_size);
      double weights4x4[3 * 4 * 4];
      PIK_RETURN_IF_ERROR(get_dct_weights(
          encoding.dct_params.distance_bands,
          encoding.dct_params.num_distance_bands,
          encoding.dct_params.eccentricity_bands,
          encoding.dct_params.num_eccentricity_bands, weights4x4));
      for (size_t c = 0; c < 3; c++) {
        for (size_t y = 0; y < kBlockDim; y++) {
          for (size_t x = 0; x < kBlockDim; x++) {
            weights[c * num + y * kBlockDim + x] =
                weights4x4[c * 16 + (y / 2) * 4 + (x / 2)];
          }
        }
        weights[c * num + 1] /= encoding.dct4multipliers[c][0];
        weights[c * num + N] /= encoding.dct4multipliers[c][0];
        weights[c * num + N + 1] /= encoding.dct4multipliers[c][1];
      }
      break;
    }
    case QuantEncoding::kQuantModeDCT: {
      PIK_RETURN_IF_ERROR(
          get_dct_weights(encoding.dct_params.distance_bands,
                          encoding.dct_params.num_distance_bands,
                          encoding.dct_params.eccentricity_bands,
                          encoding.dct_params.num_eccentricity_bands, weights));
      break;
    }
    case QuantEncoding::kQuantModeRaw: {
      PIK_ASSERT(num == encoding.block_dim * encoding.block_dim * block_size);
      for (size_t c = 0; c < 3; c++) {
        for (size_t i = 0; i < num; i++) {
          weights[c * num + i] = encoding.weights[c][i];
        }
      }
      break;
    }
    case QuantEncoding::kQuantModeRawScaled: {
      PIK_ASSERT(num == encoding.block_dim * encoding.block_dim * block_size);
      for (size_t c = 0; c < 3; c++) {
        for (size_t i = 0; i < num; i++) {
          weights[c * num + i] = encoding.weights[0][i] * encoding.scales[c];
        }
      }
      break;
    }
  }
  for (size_t c = 0; c < 3; c++) {
    offsets[kind * 3 + c] = *pos;
    for (size_t i = 0; i < num; i++) {
      double val = numerators[i] / weights[c * num + i];
      if (val > std::numeric_limits<float>::max() || val < 0) {
        return PIK_FAILURE("Invalid quantization table");
      }
      table[(*pos)++] = val;
    }
  }
  return true;
}
}  // namespace

std::string DequantMatrices::Encode(PikImageSizeInfo* info) const {
  PIK_ASSERT(encodings_.size() < std::numeric_limits<uint8_t>::max());
  uint8_t num_tables = encodings_.size();
  while (num_tables > 0 &&
         encodings_[num_tables - 1].mode == QuantEncoding::kQuantModeDefault) {
    num_tables--;
  }
  std::string out(1, num_tables);
  for (size_t i = 0; i < num_tables; i++) {
    out += pik::Encode(encodings_[i]);
  }
  if (info != nullptr) {
    info->total_size += out.size();
  }
  return out;
}

Status DequantMatrices::Decode(BitReader* br) {
  br->FillBitBuffer();
  size_t num_tables = br->ReadBits(8);
  encodings_.clear();
  if (num_tables > kNumQuantKinds) {
    Warning("Too many quantization tables: %zu > %d", num_tables,
            kNumQuantKinds);
  }
  encodings_.resize(std::max<size_t>(num_tables, kNumQuantKinds),
                    QuantEncoding::Default());
  size_t required_size[kNumQuantKinds] = {1, 1, 1, 1, 2, 4, 1};
  for (size_t i = 0; i < num_tables; i++) {
    PIK_RETURN_IF_ERROR(
        pik::Decode(br, &encodings_[i], required_size[i % kNumQuantKinds]));
  }
  static_assert(kNumQuantKinds == 7,
                "Update this function when adding new quantization kinds.");
  return DequantMatrices::Compute();
}

Status DequantMatrices::Compute() {
  size_t pos = 0;

  static_assert(kNumQuantKinds == 7,
                "Update this function when adding new quantization kinds.");

  QuantEncoding defaults[kNumQuantKinds];

  // DCT8
  {
    const float distance_bands[3][6] = {
        {7.113886368038461, -0.43371246868528784, -0.2685470152906001,
         -1.933878521411768, 0.53033146440815437, 1.4584704571124019},
        {1.6176127520379999, -0.030590539811721797, -0.81511728276358797,
         -0.67440871243431311, -0.15712359755885325, -0.73924568370505372},
        {0.54011754794716416, -6.3034056979231083, 2.3273572414645964,
         -1.6409250040790331, 4.1853947041587629, -3.7457300653769448},
    };

    const float eccentricity_bands[3][3] = {
        {-0.061872233421693838, 0.18304708783697204, -0.17934912597523417},
        {-0.046291470545973795, -0.043761762986084363, 0.13108010180056726},
        {-0.59012724924929572, 0.05090335816668784, 2.4708138138989524},
    };
    defaults[kQuantKindDCT8] = QuantEncoding::DCT(
        DctQuantWeightParams(distance_bands, eccentricity_bands));
  }

  // Identity
  {
    float weights[3][3] = {
        {289.54327306246569, 7099.418899943772, 4458.0161440926058},
        {49.176734165646614, 1499.6019634108013, 1344.3667073511192},
        {10.632431555654138, 24.895364788262089, 10.436948048081218},
    };
    defaults[kQuantKindID] =
        QuantEncoding::Identity(weights[0], weights[1], weights[2]);
  }

  // DCT2
  {
    float weights[3][6] = {
        {4577.1135330420657, 2818.0013383004025, 1244.7777050336688,
         978.20732875993258, 239.94519506075841, 153.36260218063393},
        {1083.4619643789881, 1090.5935756979763, 346.41523329464525,
         246.86085015595222, 67.994473729240639, 40.321929427434902},
        {187.56861329076196, 157.50261436421229, 97.901734624122881,
         86.611876754006317, 28.907126780373208, 19.026190235913297},
    };
    defaults[kQuantKindDCT2] =
        QuantEncoding::DCT2(weights[0], weights[1], weights[2]);
  }

  // DCT4 (quant_kind 3)
  {
    const float distance_bands[3][4] = {
        {20.174926523945018, -0.48241228169173933, -0.54668900221566374,
         -10.005340867323582},
        //
        {4.509726380079238, -0.14876246280941477, -2.2910015747473857,
         -0.11960982261055376},
        //
        {4.5461864045963649, -16.849409758229385, 1.509270337320165,
         -19.689392418307307},
    };

    const float eccentricity_bands[3][2] = {
        {-0.56774040486237598, 0.70960965891803784},
        //
        {-0.047016047231028639, 0.33992358943056145},
        //
        {-0.7776904972216967, 3.272614976356607},
    };
    const float muls[3][2] = {{0.34535223353306133, 0.51268795537431522},
                              {0.26188250178636685, 0.30589811145204077},
                              {1.7635199264413359, 1.7656127341515142}};
    defaults[kQuantKindDCT4] = QuantEncoding::DCT4(
        DctQuantWeightParams(distance_bands, eccentricity_bands), muls[0],
        muls[1], muls[2]);
  }

  // DCT16
  {
    const float distance_bands[3][6] = {
        {3.1447162136920208, -1.8170074633128082, 0.43625813680676828,
         -1.2362921454625424, -29.525222754928986, -41.379845930084585},
        {0.67135863356761205, -0.59012957157786006, -1.261084780844391,
         -0.79474653777229298, -0.58780396467734441, -1.2828836011550928},
        {0.41900092114532278, -5.5848895041895714, 1.7869039980993484,
         -6.1387179834064911, -86.733306198885529, -4.4246565408227818},
    };

    const float eccentricity_bands[3][3] = {
        {-0.043244373886759703, -0.087931295489024522, -0.29407572699703954},
        {0.080594256215966636, -0.080634363588261843, -0.091356217117333605},
        {-0.32255838560797701, -0.44093039699319492, 0.097658819835651986},
    };
    defaults[kQuantKindDCT16] = QuantEncoding::DCT(
        DctQuantWeightParams(distance_bands, eccentricity_bands));
  }

  // DCT32
  {
    const float distance_bands[3][8] = {
        {0.66392076319574023, -1.4219558051980794, 0.2981334864386988,
         0.78333090349311385, -6.707494198176172, -13.240342969589957,
         0.63851543527327559, 21.590467010691533},
        {0.20445654189211382, -0.84065253126142414, -0.59300255176397787,
         -1.0307939617135555, -0.13455419160958743, -0.26046641140749527,
         -0.48027219333962601, -1.0283737298867768},
        {0.34652699325328962, -10.648637307629331, -6.0276367959858375,
         -5.3248326849728134, -10.913916765644046, 3.3626960818075111,
         1.1749292720527955, -1.4730486958583846},
    };

    const float eccentricity_bands[3][4] = {
        {-0.51383478850903075, 0.58960854659221762, -0.48573912817312798,
         -0.18919493410017951},
        {0.089915894990142076, 0.10306681449710839, -0.25257181144076263,
         -0.070238058918648399},
        {0.73743479499112385, -1.9888390355965009, -7.830102234616005,
         0.2748609199322164},
    };
    defaults[kQuantKindDCT32] = QuantEncoding::DCT(
        DctQuantWeightParams(distance_bands, eccentricity_bands));
  }

  // Diagonal lines
  {
    static const float kPositionWeights[64] = {
        0,   100, 100, 100, 100, 100, 100, 5, 100, 100, 50, 20, 20, 10, 5, 5,
        100, 100, 50,  20,  20,  10,  5,   5, 100, 50,  50, 20, 20, 10, 5, 5,
        100, 20,  20,  20,  20,  10,  5,   5, 100, 10,  10, 10, 10, 10, 5, 5,
        100, 5,   5,   5,   5,   5,   5,   5, 5,   5,   5,  5,  5,  5,  5, 5,
    };
    static const float kChannelWeights[3] = {7.0, 17.5, 0.35};
    defaults[kQuantKindLines] =
        QuantEncoding::RawScaled(1, kPositionWeights, kChannelWeights);
  }
  for (size_t kind = 0; kind < kNumQuantKinds; kind++) {
    if (encodings_[kind].mode == QuantEncoding::kQuantModeDefault) {
      PIK_RETURN_IF_ERROR(ComputeQuantTable(
          defaults[kind], table_, table_offsets_, (QuantKind)kind, &pos));
    } else {
      PIK_RETURN_IF_ERROR(ComputeQuantTable(
          encodings_[kind], table_, table_offsets_, (QuantKind)kind, &pos));
    }
  }

  size_ = pos;
  if (need_inv_matrices_) {
    for (size_t i = 0; i < pos; i++) {
      inv_table_[i] = 1.0f / table_[i];
    }
  }
  return true;
}

DequantMatrices FindBestDequantMatrices(float butteraugli_target,
                                        const Image3F& opsin) {
  // TODO(veluca): heuristics for in-bitstream quant tables.
  return DequantMatrices(/*need_inv_matrices=*/true);
}

}  // namespace pik
