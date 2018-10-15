// Copyright 2018 Google Inc. All Rights Reserved.
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

#include "color_management.h"

#include <mutex>
#include "third_party/lcms/include/lcms2.h"

#include "rational_polynomial.h"
#include "simd/simd.h"

namespace pik {
namespace {

#define PIK_CMS_VERBOSE 0

// Protects non-thread-safe functions.
std::mutex lcms_mutex;

// (LCMS interface requires xyY but we omit the Y for white points/primaries.)

PIK_MUST_USE_RESULT CIExy CIExyFromxyY(const cmsCIExyY& xyY) {
  const CIExy xy = {xyY.x, xyY.y};
  return xy;
}

PIK_MUST_USE_RESULT cmsCIExyY xyYFromCIExy(const CIExy& xy) {
  const cmsCIExyY xyY = {xy.x, xy.y, 1.0};
  return xyY;
}

PIK_MUST_USE_RESULT CIExy CIExyFromXYZ(const cmsCIEXYZ& XYZ) {
  cmsCIExyY xyY;
  cmsXYZ2xyY(/*Dest=*/&xyY, /*Source=*/&XYZ);
  return CIExyFromxyY(xyY);
}

PIK_MUST_USE_RESULT cmsCIEXYZ D50_XYZ() {
  // Quantized D50 as stored in ICC profiles.
  return {0.96420288, 1.0, 0.82490540};
}

// RAII

struct ProfileDeleter {
  void operator()(void* p) { cmsCloseProfile(p); }
};
using Profile = std::unique_ptr<void, ProfileDeleter>;

Status CreateProfileXYZ(const cmsContext context,
                        Profile* PIK_RESTRICT profile) {
  // Data-race in cmsD50_xyY (modifies static variable).
  lcms_mutex.lock();
  profile->reset(cmsCreateXYZProfileTHR(context));
  lcms_mutex.unlock();
  if (profile->get() == nullptr) return PIK_FAILURE("Failed to create XYZ");
  return true;
}

// Multi-Localized Unicode string
class MLU {
 public:
  MLU(const cmsContext context, const char* ascii)
      : mlu_(cmsMLUalloc(context, 0)) {
    if (!cmsMLUsetASCII(mlu_, "en", "US", ascii)) {
      PIK_NOTIFY_ERROR("Failed to set ASCII");
    }
  }
  ~MLU() { cmsMLUfree(mlu_); }

  MLU(const MLU&) = delete;
  MLU& operator=(const MLU&) = delete;
  MLU(MLU&&) = delete;
  MLU& operator=(MLU&&) = delete;

  cmsMLU* get() const { return mlu_; }

 private:
  cmsMLU* mlu_;
};

// Sets header and required tags; called by EncodeProfile.
Status SetTags(const cmsContext context, const Profile& profile,
               const std::string& profile_description) {
  cmsHPROFILE p = profile.get();

  // Header
  cmsSetHeaderFlags(p, 1);  // embedded

  const MLU copyright(
      context,
      "Copyright 2018 Google LLC, CC-BY-SA 3.0 Unported license"
      "(https://creativecommons.org/licenses/by-sa/3.0/legalcode)");
  const MLU manufacturer(context, "Google");
  const MLU model(context, "Image codec");
  const MLU description(context, profile_description.c_str());

  // Required tags
  bool all_ok = true;
  all_ok &= cmsWriteTag(p, cmsSigCopyrightTag, copyright.get());
  all_ok &= cmsWriteTag(p, cmsSigDeviceMfgDescTag, manufacturer.get());
  all_ok &= cmsWriteTag(p, cmsSigDeviceModelDescTag, model.get());
  all_ok &= cmsWriteTag(p, cmsSigProfileDescriptionTag, description.get());

  all_ok &= cmsMD5computeID(p);
  if (!all_ok) return PIK_FAILURE("Failed to write header/tags");
  return true;
}

Status EncodeProfile(const cmsContext context, const Profile& profile,
                     const std::string& description, PaddedBytes* icc) {
  PIK_RETURN_IF_ERROR(SetTags(context, profile, description));

  cmsUInt32Number size;
  if (!cmsSaveProfileToMem(profile.get(), nullptr, &size)) {
    return PIK_FAILURE("Failed to get profile size");
  }

  icc->resize(size);
  if (!cmsSaveProfileToMem(profile.get(), icc->data(), &size)) {
    return PIK_FAILURE("Failed to encode profile");
  }
  return true;
}

Status DecodeProfile(const cmsContext context, const PaddedBytes& icc,
                     Profile* profile) {
  lcms_mutex.lock();  // gmtime in cmsCreateProfilePlaceholder
  profile->reset(cmsOpenProfileFromMemTHR(context, icc.data(), icc.size()));
  lcms_mutex.unlock();
  if (profile->get() == nullptr) {
    return PIK_FAILURE("Failed to decode profile");
  }

  // Try to detect corrupt profiles by verifying MD5.
  uint8_t md5_actual[16];
  cmsGetHeaderProfileID(profile->get(), md5_actual);
  // Only possible if the signature was previously computed.
  const uint8_t md5_zero[16] = {0};
  if (memcmp(md5_actual, md5_zero, 16) != 0) {
    if (cmsMD5computeID(profile->get())) {
      uint8_t md5_expected[16];
      cmsGetHeaderProfileID(profile->get(), md5_expected);
      if (memcmp(md5_expected, md5_actual, 16) != 0) {
        profile->reset();
        return PIK_FAILURE("MD5 mismatch, ignoring profile");
      }
    } else {
      return PIK_FAILURE("Failed to compute profile MD5");
    }
  }

  return true;
}

struct CurveDeleter {
  void operator()(cmsToneCurve* p) { cmsFreeToneCurve(p); }
};
using Curve = std::unique_ptr<cmsToneCurve, CurveDeleter>;

// Definitions for BT.2100-2 transfer functions:
// "display" is linear light (nits) normalized to [0, 1].
// "encoded" is a nonlinear encoding (e.g. PQ) in [0, 1].
// "scene" is a linear function of photon counts, normalized to [0, 1].

// Despite the stated ranges, we need unbounded transfer functions: see
// http://www.littlecms.com/CIC18_UnboundedCMM.pdf. Inputs can be negative or
// above 1 due to chromatic adaptation. To avoid severe round-trip errors caused
// by clamping, we mirror negative inputs via copysign (f(-x) = -f(x), see
// https://developer.apple.com/documentation/coregraphics/cgcolorspace/1644735-extendedsrgb)
// and extend the function domains above 1.

// Hybrid Log-Gamma.
class TF_HLG {
 public:
  // EOTF. e = encoded.
  PIK_INLINE double DisplayFromEncoded(const double e) const {
    const double lifted = e * (1.0 - kBeta) + kBeta;
    return OOTF(InvOETF(lifted));
  }

  // Inverse EOTF. d = display.
  PIK_INLINE double EncodedFromDisplay(const double d) const {
    const double lifted = OETF(InvOOTF(d));
    const double e = (lifted - kBeta) * (1.0 / (1.0 - kBeta));
    return e;
  }

 private:
  // OETF (defines the HLG approach). s = scene, returns encoded.
  PIK_INLINE double OETF(double s) const {
    if (s == 0.0) return 0.0;
    const double original_sign = s;
    s = std::abs(s);

    if (s <= kDiv12) return std::copysign(std::sqrt(3.0 * s), original_sign);

    const double e = kA * std::log(12 * s - kB) + kC;
    PIK_ASSERT(e > 0.0);
    return std::copysign(e, original_sign);
  }

  // e = encoded, returns scene.
  PIK_INLINE double InvOETF(double e) const {
    if (e == 0.0) return 0.0;
    const double original_sign = e;
    e = std::abs(e);

    if (e <= 0.5) return std::copysign(e * e * (1.0 / 3), original_sign);

    const double s = (std::exp((e - kC) * kRA) + kB) * kDiv12;
    PIK_ASSERT(s >= 0);
    return std::copysign(s, original_sign);
  }

  // s = scene, returns display.
  PIK_INLINE double OOTF(const double s) const {
    // The actual (red channel) OOTF is RD = alpha * YS^(gamma-1) * RS, where
    // YS = 0.2627 * RS + 0.6780 * GS + 0.0593 * BS. Let alpha = 1 so we return
    // "display" (normalized [0, 1]) instead of nits. Our transfer function
    // interface does not allow a dependency on YS. Fortunately, the system
    // gamma at 334 nits is 1.0, so this reduces to RD = RS.
    return s;
  }

  // d = display, returns scene.
  PIK_INLINE double InvOOTF(const double d) const {
    return d;  // see OOTF().
  }

  // Assume 1000:1 contrast @ 200 nits => gamma 0.9
  static constexpr double kBeta = 0.04;  // = sqrt(3 * contrast^(1/gamma))

  static constexpr double kA = 0.17883277;
  static constexpr double kRA = 1.0 / kA;
  static constexpr double kB = 1 - 4 * kA;
  static constexpr double kC = 0.5599107295;
  static constexpr double kDiv12 = 1.0 / 12;
};

// Perceptual Quantization
class TF_PQ {
 public:
  // EOTF (defines the PQ approach). e = encoded.
  PIK_INLINE double DisplayFromEncoded(double e) const {
    if (e == 0.0) return 0.0;
    const double original_sign = e;
    e = std::abs(e);

    const double xp = std::pow(e, 1.0 / kM2);
    const double num = std::max(xp - kC1, 0.0);
    const double den = kC2 - kC3 * xp;
    PIK_ASSERT(den != 0.0);
    const double d = std::pow(num / den, 1.0 / kM1);
    PIK_ASSERT(d >= 0.0);  // Equal for e ~= 1E-9
    return std::copysign(d, original_sign);
  }

  // Inverse EOTF. d = display.
  PIK_INLINE double EncodedFromDisplay(double d) const {
    if (d == 0.0) return 0.0;
    const double original_sign = d;
    d = std::abs(d);

    const double xp = std::pow(d, kM1);
    const double num = kC1 + xp * kC2;
    const double den = 1.0 + xp * kC3;
    const double e = std::pow(num / den, kM2);
    PIK_ASSERT(e > 0.0);
    return std::copysign(e, original_sign);
  }

 private:
  static constexpr double kM1 = 2610.0 / 16384;
  static constexpr double kM2 = (2523.0 / 4096) * 128;
  static constexpr double kC1 = 3424.0 / 4096;
  static constexpr double kC2 = (2413.0 / 4096) * 32;
  static constexpr double kC3 = (2392.0 / 4096) * 32;
};

// sRGB
class TF_SRGB {
 public:
  template <typename V>
  SIMD_ATTR PIK_INLINE V DisplayFromEncoded(V x) const {
    // Computed via af_cheb_rational (k=100); replicated 4x.
    SIMD_ALIGN constexpr float p[(4 + 1) * 4] = {
        2.200248328e-04, 2.200248328e-04, 2.200248328e-04, 2.200248328e-04,
        1.043637593e-02, 1.043637593e-02, 1.043637593e-02, 1.043637593e-02,
        1.624820318e-01, 1.624820318e-01, 1.624820318e-01, 1.624820318e-01,
        7.961564959e-01, 7.961564959e-01, 7.961564959e-01, 7.961564959e-01,
        8.210152774e-01, 8.210152774e-01, 8.210152774e-01, 8.210152774e-01,
    };
    SIMD_ALIGN constexpr float q[(4 + 1) * 4] = {
        2.631846970e-01,  2.631846970e-01,  2.631846970e-01,  2.631846970e-01,
        1.076976492e+00,  1.076976492e+00,  1.076976492e+00,  1.076976492e+00,
        4.987528350e-01,  4.987528350e-01,  4.987528350e-01,  4.987528350e-01,
        -5.512498495e-02, -5.512498495e-02, -5.512498495e-02, -5.512498495e-02,
        6.521209011e-03,  6.521209011e-03,  6.521209011e-03,  6.521209011e-03,
    };
    const SIMD_FULL(float) d;
    const V linear = x * set1(d, kLowDivInv);
    const V poly = EvalRationalPolynomial(x, p, q);
    return select(linear, poly, x > set1(d, kThreshSRGBToLinear));
  }

  template <class V>
  SIMD_ATTR PIK_INLINE V EncodedFromDisplay(const V x) const {
    // Computed via af_cheb_rational (k=100); replicated 4x.
    SIMD_ALIGN constexpr float p[(4 + 1) * 4] = {
        -5.135152395e-04, -5.135152395e-04, -5.135152395e-04, -5.135152395e-04,
        5.287254571e-03,  5.287254571e-03,  5.287254571e-03,  5.287254571e-03,
        3.903842876e-01,  3.903842876e-01,  3.903842876e-01,  3.903842876e-01,
        1.474205315e+00,  1.474205315e+00,  1.474205315e+00,  1.474205315e+00,
        7.352629620e-01,  7.352629620e-01,  7.352629620e-01,  7.352629620e-01,
    };
    SIMD_ALIGN constexpr float q[(4 + 1) * 4] = {
        1.004519624e-02, 1.004519624e-02, 1.004519624e-02, 1.004519624e-02,
        3.036675394e-01, 3.036675394e-01, 3.036675394e-01, 3.036675394e-01,
        1.340816930e+00, 1.340816930e+00, 1.340816930e+00, 1.340816930e+00,
        9.258482155e-01, 9.258482155e-01, 9.258482155e-01, 9.258482155e-01,
        2.424867759e-02, 2.424867759e-02, 2.424867759e-02, 2.424867759e-02,
    };
    const SIMD_FULL(float) d;
    const V linear = x * set1(d, kLowDiv);
    const V poly = EvalRationalPolynomial(sqrt(x), p, q);
    return select(linear, poly, x > set1(d, kThreshLinearToSRGB));
  }

 private:
  static constexpr double kThreshSRGBToLinear = 0.04045;
  static constexpr double kThreshLinearToSRGB = 0.0031308;
  static constexpr double kLowDiv = 12.92;
  static constexpr double kLowDivInv = 1.0 / kLowDiv;
};

// NOTE: this is only used to provide a reasonable ICC profile that other
// software can read. Our own transforms use ExtraTF instead because that is
// more precise and supports unbounded mode.
template <class Func>
cmsToneCurve* CreateTableCurve(const cmsContext context, int32_t N,
                               const Func& func) {
  PIK_ASSERT(N <= 4096);  // ICC MFT2 only allows 4K entries
  // No point using float - LCMS converts to 16-bit for A2B/MFT.
  std::vector<uint16_t> table;
  table.reserve(N);
  for (int32_t i = 0; i < N; ++i) {
    const float x = static_cast<float>(i) / (N - 1);  // 1.0 at index N - 1.
    // LCMS requires EOTF (e.g. 2.4 exponent).
    float y = func.DisplayFromEncoded(x);
    PIK_ASSERT(y >= 0.0f);
    // Clamp to table range - necessary for HLG.
    if (y > 1.0f) y = 1.0f;
    table.push_back(std::round(y * 65535.0f));  // 1.0 at table value 0xFFFF.
  }
  return cmsBuildTabulatedToneCurve16(context, N, table.data());
}

Curve CreateCurve(const cmsContext context, const double gamma) {
  const cmsUInt32Number type = 4;  // exponential with linear part

  PIK_CHECK(0 < gamma && gamma <= 1.0);

  if (ApproxEq(gamma, GammaSRGB())) {
    constexpr cmsFloat64Number params[5] = {2.4, 1.0 / 1.055, 0.055 / 1.055,
                                            1.0 / 12.92, 0.04045};
    return Curve(cmsBuildParametricToneCurve(context, type, params));
  } else if (ApproxEq(gamma, Gamma709())) {
    constexpr cmsFloat64Number params[5] = {1.0 / 0.45, 1.0 / 1.099,
                                            0.099 / 1.099, 1.0 / 4.5, 0.081};
    return Curve(cmsBuildParametricToneCurve(context, type, params));
  } else if (ApproxEq(gamma, GammaHLG())) {
    return Curve(CreateTableCurve(context, 4096, TF_HLG()));
  } else if (ApproxEq(gamma, GammaPQ())) {
    return Curve(CreateTableCurve(context, 4096, TF_PQ()));
  } else {
    // "gamma" is the OETF exponent; LCMS expects EOTF, so take the reciprocal.
    return Curve(cmsBuildGamma(context, 1.0 / gamma));
  }
}

// Returns false for unsupported color_space and gamma (not an error).
// Serializes the profile before use to ensure all values are quantized.
Status MaybeCreateProfile(const cmsContext context, const ProfileParams& pp,
                          PaddedBytes* PIK_RESTRICT icc) {
  if (pp.gamma == 0.0) return false;  // Unknown gamma, not an error.

  // (If color_space == kRGB, we'll use this curve for all channels.)
  const Curve curve = CreateCurve(context, pp.gamma);
  if (curve == nullptr) return PIK_FAILURE("Failed to create curve");

  const cmsCIExyY wp_xyY = xyYFromCIExy(pp.white_point);

  Profile profile;
  if (pp.color_space == ColorSpace::kRGB) {
    const cmsCIExyYTRIPLE primaries_xyY = {xyYFromCIExy(pp.primaries.r),
                                           xyYFromCIExy(pp.primaries.g),
                                           xyYFromCIExy(pp.primaries.b)};
    cmsToneCurve* curves[3] = {curve.get(), curve.get(), curve.get()};
    lcms_mutex.lock();  // gmtime in cmsCreateProfilePlaceholder
    profile.reset(
        cmsCreateRGBProfileTHR(context, &wp_xyY, &primaries_xyY, curves));
    lcms_mutex.unlock();
    if (profile.get() == nullptr) return PIK_FAILURE("Failed to create RGB");
  } else if (pp.color_space == ColorSpace::kGray) {
    lcms_mutex.lock();  // gmtime in cmsCreateProfilePlaceholder
    profile.reset(cmsCreateGrayProfileTHR(context, &wp_xyY, curve.get()));
    lcms_mutex.unlock();
    if (profile.get() == nullptr) return PIK_FAILURE("Failed to create Gray");
  } else if (pp.color_space == ColorSpace::kXYZ) {
    PIK_RETURN_IF_ERROR(CreateProfileXYZ(context, &profile));  // takes lock
  } else {
    return false;  // not an error. TODO(janwas): handle others
  }

  // ICC uses the same values.
  cmsSetHeaderRenderingIntent(
      profile.get(), static_cast<cmsUInt32Number>(pp.rendering_intent));

  return EncodeProfile(context, profile, Description(pp), icc);
}

uint32_t Type32(const ColorEncoding& c) {
  if (c.IsGray()) return TYPE_GRAY_FLT;
  if (c.color_space == ColorSpace::kXYZ) return TYPE_XYZ_FLT;
  return TYPE_RGB_FLT;
}

uint32_t Type64(const ColorEncoding& c) {
  if (c.IsGray()) return TYPE_GRAY_DBL;
  if (c.color_space == ColorSpace::kXYZ) return TYPE_XYZ_DBL;
  return TYPE_RGB_DBL;
}

PIK_MUST_USE_RESULT ColorSpace ColorSpaceFromProfile(const Profile& profile) {
  switch (cmsGetColorSpace(profile.get())) {
    case cmsSigRgbData:
      return ColorSpace::kRGB;
    case cmsSigGrayData:
      return ColorSpace::kGray;
    case cmsSigXYZData:
      return ColorSpace::kXYZ;
    case cmsSigYCbCrData:
      return ColorSpace::kYCbCr;
    default:
      return ColorSpace::kUnknown;
  }
}

// "profile1" is pre-decoded to save time in DetectTransferFunction.
Status ProfileEquivalentToICC(const cmsContext context, const Profile& profile1,
                              const PaddedBytes& icc, const ColorEncoding& c) {
  const uint32_t type_src = Type64(c);

  Profile profile2;
  PIK_RETURN_IF_ERROR(DecodeProfile(context, icc, &profile2));

  Profile profile_xyz;
  PIK_RETURN_IF_ERROR(CreateProfileXYZ(context, &profile_xyz));

  const uint32_t intent = INTENT_RELATIVE_COLORIMETRIC;
  const uint32_t flags = cmsFLAGS_NOOPTIMIZE | cmsFLAGS_BLACKPOINTCOMPENSATION |
                         cmsFLAGS_HIGHRESPRECALC;
  lcms_mutex.lock();  // cmsD50_xyY race
  ColorManagement::Transform xform1(
      cmsCreateTransformTHR(context, profile1.get(), type_src,
                            profile_xyz.get(), TYPE_XYZ_DBL, intent, flags));
  ColorManagement::Transform xform2(
      cmsCreateTransformTHR(context, profile2.get(), type_src,
                            profile_xyz.get(), TYPE_XYZ_DBL, intent, flags));
  lcms_mutex.unlock();
  if (xform1 == nullptr || xform2 == nullptr) {
    return PIK_FAILURE("Failed to create transform");
  }

  double in[3];
  double out1[3];
  double out2[3];

  // Uniformly spaced samples from very dark to almost fully bright.
  const double init = 1E-3;
  const double step = 0.2;

  if (c.IsGray()) {
    // Finer sampling and replicate each component.
    for (in [0] = init; in[0] < 1.0; in[0] += step / 8) {
      cmsDoTransform(xform1.get(), in, out1, 1);
      cmsDoTransform(xform2.get(), in, out2, 1);
      if (!ApproxEq(out1[0], out2[0])) {
        return false;
      }
    }
  } else {
    for (in [0] = init; in[0] < 1.0; in[0] += step) {
      for (in [1] = init; in[1] < 1.0; in[1] += step) {
        for (in [2] = init; in[2] < 1.0; in[2] += step) {
          cmsDoTransform(xform1.get(), in, out1, 1);
          cmsDoTransform(xform2.get(), in, out2, 1);
          for (size_t i = 0; i < 3; ++i) {
            if (!ApproxEq(out1[i], out2[i])) {
              return false;
            }
          }
        }
      }
    }
  }

  return true;
}

// Returns white point that was specified when creating the profile.
// NOTE: we can't just use cmsSigMediaWhitePointTag because its interpretation
// differs between ICC versions.
PIK_MUST_USE_RESULT cmsCIEXYZ UnadaptedWhitePoint(const cmsContext context,
                                                  const Profile& profile,
                                                  const ColorEncoding& c) {
  cmsCIEXYZ XYZ = {1.0, 1.0, 1.0};

  Profile profile_xyz;
  if (!CreateProfileXYZ(context, &profile_xyz)) return XYZ;
  // Array arguments are one per profile.
  cmsHPROFILE profiles[2] = {profile.get(), profile_xyz.get()};
  // Leave white point unchanged - that is what we're trying to extract.
  cmsUInt32Number intents[2] = {INTENT_ABSOLUTE_COLORIMETRIC,
                                INTENT_ABSOLUTE_COLORIMETRIC};
  cmsBool black_compensation[2] = {0, 0};
  cmsFloat64Number adaption[2] = {0.0, 0.0};
  // Only transforming a single pixel, so skip expensive optimizations.
  cmsUInt32Number flags = cmsFLAGS_NOOPTIMIZE | cmsFLAGS_HIGHRESPRECALC;
  lcms_mutex.lock();  // cmsD50_xyY race
  ColorManagement::Transform xform(cmsCreateExtendedTransform(
      context, 2, profiles, black_compensation, intents, adaption, nullptr, 0,
      Type64(c), TYPE_XYZ_DBL, flags));
  lcms_mutex.unlock();

  // xy are relative, so magnitude does not matter if we ignore output Y.
  const cmsFloat64Number in[3] = {1.0, 1.0, 1.0};
  cmsDoTransform(xform.get(), in, &XYZ.X, 1);
  return XYZ;
}

PIK_MUST_USE_RESULT Primaries IdentifyPrimaries(const Profile& profile,
                                                const cmsCIEXYZ& wp_unadapted) {
  // These were adapted to the profile illuminant before storing in the profile.
  const cmsCIEXYZ* adapted_r = static_cast<const cmsCIEXYZ*>(
      cmsReadTag(profile.get(), cmsSigRedColorantTag));
  const cmsCIEXYZ* adapted_g = static_cast<const cmsCIEXYZ*>(
      cmsReadTag(profile.get(), cmsSigGreenColorantTag));
  const cmsCIEXYZ* adapted_b = static_cast<const cmsCIEXYZ*>(
      cmsReadTag(profile.get(), cmsSigBlueColorantTag));
  if (adapted_r == nullptr || adapted_g == nullptr || adapted_b == nullptr) {
    PIK_NOTIFY_ERROR("Failed to retrieve colorants");
    return Primaries::kUnknown;
  }

  // TODO(janwas): no longer assume Bradford and D50.
  // Undo the chromatic adaptation.
  const cmsCIEXYZ d50 = D50_XYZ();

  cmsCIEXYZ r, g, b;
  cmsAdaptToIlluminant(&r, &d50, &wp_unadapted, adapted_r);
  cmsAdaptToIlluminant(&g, &d50, &wp_unadapted, adapted_g);
  cmsAdaptToIlluminant(&b, &d50, &wp_unadapted, adapted_b);

  const PrimariesCIExy rgb = {CIExyFromXYZ(r), CIExyFromXYZ(g),
                              CIExyFromXYZ(b)};
  return PrimariesFromCIExy(rgb);
}

PIK_MUST_USE_RESULT TransferFunction DetectTransferFunction(
    const cmsContext context, ColorEncoding* PIK_RESTRICT c) {
  ProfileParams pp = ParamsFromColorEncoding(*c);

  Profile profile;
  if (!DecodeProfile(context, c->icc, &profile)) {
    return TransferFunction::kUnknown;
  }

  for (TransferFunction tf : AllValues<TransferFunction>()) {
    pp.gamma = GammaFromTransferFunction(tf);

    PaddedBytes icc_test;
    if (MaybeCreateProfile(context, pp, &icc_test) &&
        ProfileEquivalentToICC(context, profile, icc_test, *c)) {
      return tf;
    }
  }

  return TransferFunction::kUnknown;
}

}  // namespace

void ColorManagement::ContextDeleter::operator()(void* p) {
  cmsDeleteContext(static_cast<cmsContext>(p));
}

void ColorManagement::TransformDeleter::operator()(void* p) {
  cmsDeleteTransform(p);
}

ColorManagement::ColorManagement(size_t num_threads) {
  PIK_CHECK(LCMS_VERSION == cmsGetEncodedCMMversion());

  struct ErrorHandler {
    static void Run(cmsContext context, cmsUInt32Number code,
                    const char* text) {
      fprintf(stderr, "LCMS error %u: %s\n", code, text);
    }
  };

  const size_t num_contexts = std::max<size_t>(num_threads, 1);
  contexts_.reserve(num_contexts);
  for (size_t i = 0; i < num_contexts; ++i) {
    const cmsContext context = cmsCreateContext(nullptr, nullptr);
    contexts_.emplace_back(context);
    cmsSetLogErrorHandlerTHR(context, &ErrorHandler::Run);
  }
}

Status ColorManagement::SetFromParams(const ProfileParams& pp,
                                      ColorEncoding* PIK_RESTRICT c) const {
  const auto context = static_cast<cmsContext>(contexts_[0].get());
  if (!MaybeCreateProfile(context, pp, &c->icc)) {
    return PIK_FAILURE("Failed to create profile");
  }
  SetFieldsFromParams(pp, c);
  return true;
}

Status ColorManagement::SetFromProfile(PaddedBytes&& icc,
                                       ColorEncoding* PIK_RESTRICT c) const {
  if (icc.empty()) return false;
  const auto context = static_cast<cmsContext>(contexts_[0].get());

  Profile profile;
  PIK_RETURN_IF_ERROR(DecodeProfile(context, icc, &profile));

  c->icc = std::move(icc);

  c->color_space = ColorSpaceFromProfile(profile);

  const cmsCIEXYZ wp_unadapted = UnadaptedWhitePoint(context, profile, *c);
  c->white_point = WhitePointFromCIExy(CIExyFromXYZ(wp_unadapted));

  // Gray/XYZ profiles don't have primaries.
  c->primaries = Primaries::kUnknown;
  if (c->color_space != ColorSpace::kGray &&
      c->color_space != ColorSpace::kXYZ) {
    c->primaries = IdentifyPrimaries(profile, wp_unadapted);
  }

  // XYZ profiles are always linear.
  c->transfer_function = TransferFunction::kLinear;
  if (c->color_space != ColorSpace::kXYZ) {
    // Must come last because it uses the other fields.
    c->transfer_function = DetectTransferFunction(context, c);
  }

  // ICC uses the same values.
  c->rendering_intent =
      static_cast<RenderingIntent>(cmsGetHeaderRenderingIntent(profile.get()));

  return true;
}

Status ColorManagement::SetProfileFromFields(
    ColorEncoding* PIK_RESTRICT c) const {
  c->icc.clear();
  const auto context = static_cast<cmsContext>(contexts_[0].get());
  const ProfileParams pp = ParamsFromColorEncoding(*c);
  if (!MaybeCreateProfile(context, pp, &c->icc)) {
    return PIK_FAILURE("Failed to create profile from fields");
  }
  return true;
}

Status ColorManagement::MaybeRemoveProfile(
    ColorEncoding* PIK_RESTRICT c) const {
  // Avoid printing an error message when there is no ICC profile.
  if (c->icc.empty()) return true;
  const auto context = static_cast<cmsContext>(contexts_[0].get());

  Profile profile_old;
  PIK_RETURN_IF_ERROR(DecodeProfile(context, c->icc, &profile_old));

  const ProfileParams pp = ParamsFromColorEncoding(*c);
  PaddedBytes icc_new;
  PIK_RETURN_IF_ERROR(MaybeCreateProfile(context, pp, &icc_new));

  if (!ProfileEquivalentToICC(context, profile_old, icc_new, *c)) {
    return PIK_FAILURE("Generated profile does not match");
  }

  c->icc.clear();
  return true;
}

Status ColorSpaceTransform::Init(
    const std::vector<ColorManagement::Context>& contexts,
    const ColorEncoding& c_src, const ColorEncoding& c_dst, size_t xsize) {
  Profile profile_src, profile_dst;
  const auto context = static_cast<cmsContext>(contexts.front().get());
  PIK_RETURN_IF_ERROR(DecodeProfile(context, c_src.icc, &profile_src));
  PIK_RETURN_IF_ERROR(DecodeProfile(context, c_dst.icc, &profile_dst));
  skip_lcms_ = false;
  if (c_src.SameColorSpace(c_dst) &&
      c_src.transfer_function == c_dst.transfer_function) {
    skip_lcms_ = true;
  }

  // Special-case for BT.2100 HLG/PQ and SRGB <=> linear:
  if ((Is2100(c_src.transfer_function) && IsLinear(c_dst.transfer_function)) ||
      (Is2100(c_dst.transfer_function) && IsLinear(c_src.transfer_function)) ||
      (IsSRGB(c_src.transfer_function) && IsLinear(c_dst.transfer_function)) ||
      (IsSRGB(c_dst.transfer_function) && IsLinear(c_src.transfer_function))) {
    // Construct new profiles as if the data were already/still linear.
    ProfileParams pp_src = ParamsFromColorEncoding(c_src);
    ProfileParams pp_dst = ParamsFromColorEncoding(c_dst);
    pp_src.gamma = pp_dst.gamma = GammaLinear();
    PaddedBytes icc_src, icc_dst;
    Profile new_src, new_dst;
    // Only enable ExtraTF if profile creation succeeded.
    if (MaybeCreateProfile(context, pp_src, &icc_src) &&
        MaybeCreateProfile(context, pp_dst, &icc_dst) &&
        DecodeProfile(context, icc_src, &new_src) &&
        DecodeProfile(context, icc_dst, &new_dst)) {
      if (c_src.SameColorSpace(c_dst)) {
        skip_lcms_ = true;
      }
      profile_src.swap(new_src);
      profile_dst.swap(new_dst);
      if (IsLinear(c_dst.transfer_function)) {
        preprocess_ = IsSRGB(c_src.transfer_function)
                          ? ExtraTF::kSRGB
                          : (IsPQ(c_src.transfer_function) ? ExtraTF::kPQ
                                                           : ExtraTF::kHLG);
      } else {
        PIK_ASSERT(IsLinear(c_src.transfer_function));
        postprocess_ = IsSRGB(c_dst.transfer_function)
                           ? ExtraTF::kSRGB
                           : (IsPQ(c_dst.transfer_function) ? ExtraTF::kPQ
                                                            : ExtraTF::kHLG);
      }
    } else {
      fprintf(stderr, "Failed to create extra linear profiles");
    }
  }

  // Type includes color space (XYZ vs RGB), so can be different.
  const uint32_t type_src = Type32(c_src);
  const uint32_t type_dst = Type32(c_dst);
  // Not including alpha channel (copied separately).
  const size_t channels_src = c_src.Channels();
  const size_t channels_dst = c_dst.Channels();
  PIK_CHECK(channels_src == channels_dst);

  transforms_.clear();
  for (size_t i = 0; i < contexts.size(); ++i) {
    const auto context = static_cast<cmsContext>(contexts[i].get());
    const uint32_t intent = static_cast<uint32_t>(c_dst.rendering_intent);
    const uint32_t flags =
        cmsFLAGS_BLACKPOINTCOMPENSATION | cmsFLAGS_HIGHRESPRECALC;
    lcms_mutex.lock();  // cmsD50_xyY race
    transforms_.emplace_back(cmsCreateTransformTHR(context, profile_src.get(),
                                                   type_src, profile_dst.get(),
                                                   type_dst, intent, flags));
    lcms_mutex.unlock();
    if (transforms_.back() == nullptr) {
      return PIK_FAILURE("Failed to create transform");
    }
  }

  // Ideally LCMS would convert directly from External to Image3. However,
  // cmsDoTransformLineStride only accepts 32-bit BytesPerPlaneIn, whereas our
  // planes can be more than 4 GiB apart. Hence, transform inputs/outputs must
  // be interleaved. Calling cmsDoTransform for each pixel is expensive
  // (indirect call). We therefore transform rows, which requires per-thread
  // buffers. To avoid separate allocations, we use the rows of an image.
  // Because LCMS apparently also cannot handle <= 16 bit inputs and 32-bit
  // outputs (or vice versa), we use floating point input/output.
  buf_src_ = ImageF(xsize * channels_src, contexts.size());
  buf_dst_ = ImageF(xsize * channels_dst, contexts.size());
  xsize_ = xsize;
  return true;
}

SIMD_ATTR void ColorSpaceTransform::Run(const size_t thread,
                                        const float* buf_src, float* buf_dst) {
#if PIK_CMS_VERBOSE
  const size_t kX = 0;
#endif

  // ExtraTF can't write to (pointer-to-const) buf_src, so use buffer.
  float* xform_src = buf_src_.Row(thread);  // possibly aliases buf_src
  switch (preprocess_) {
    case ExtraTF::kNone:
      xform_src = const_cast<float*>(buf_src);  // won't write to it
      break;
    case ExtraTF::kPQ:
      for (size_t i = 0; i < buf_src_.xsize(); ++i) {
        xform_src[i] = TF_PQ().DisplayFromEncoded(buf_src[i]);
      }
#if PIK_CMS_VERBOSE
      printf("pre in %.4f %.4f %.4f undoPQ %.4f %.4f %.4f\n", buf_src[3 * kX],
             buf_src[3 * kX + 1], buf_src[3 * kX + 2], xform_src[3 * kX],
             xform_src[3 * kX + 1], xform_src[3 * kX + 2]);
#endif
      break;
    case ExtraTF::kHLG:
      for (size_t i = 0; i < buf_src_.xsize(); ++i) {
        xform_src[i] = TF_HLG().DisplayFromEncoded(buf_src[i]);
      }
#if PIK_CMS_VERBOSE
      printf("pre in %.4f %.4f %.4f undoHLG %.4f %.4f %.4f\n", buf_src[3 * kX],
             buf_src[3 * kX + 1], buf_src[3 * kX + 2], xform_src[3 * kX],
             xform_src[3 * kX + 1], xform_src[3 * kX + 2]);
#endif
      break;
    case ExtraTF::kSRGB:
      SIMD_FULL(float) df;
      for (size_t i = 0; i < buf_src_.xsize(); i += df.N) {
        const auto val = load(df, buf_src + i);
        const auto result = TF_SRGB().DisplayFromEncoded(val);
        store(result, df, xform_src + i);
      }
      break;
  }

  if (!skip_lcms_) {
    PIK_ASSERT(thread < transforms_.size());
    cmsHTRANSFORM xform = transforms_[thread].get();
    cmsDoTransform(xform, xform_src, buf_dst, xsize_);
  } else {
    memcpy(buf_dst, xform_src, buf_dst_.xsize() * sizeof(*buf_dst));
  }
#if PIK_CMS_VERBOSE
  printf("xform: %.4f %.4f %.4f (%p) -> (%p) %.4f %.4f %.4f\n",
         xform_src[3 * kX], xform_src[3 * kX + 1], xform_src[3 * kX + 2],
         xform_src, buf_dst, buf_dst[3 * kX], buf_dst[3 * kX + 1],
         buf_dst[3 * kX + 2]);
#endif

  switch (postprocess_) {
    case ExtraTF::kNone:
      break;
    case ExtraTF::kPQ:
      for (size_t i = 0; i < buf_dst_.xsize(); ++i) {
        buf_dst[i] = TF_PQ().EncodedFromDisplay(buf_dst[i]);
      }
#if PIK_CMS_VERBOSE
      printf("after PQ enc %.4f %.4f %.4f\n", buf_dst[3 * kX],
             buf_dst[3 * kX + 1], buf_dst[3 * kX + 2]);
#endif
      break;
    case ExtraTF::kHLG:
      for (size_t i = 0; i < buf_dst_.xsize(); ++i) {
        buf_dst[i] = TF_HLG().EncodedFromDisplay(buf_dst[i]);
      }
#if PIK_CMS_VERBOSE
      printf("after HLG enc %.4f %.4f %.4f\n", buf_dst[3 * kX],
             buf_dst[3 * kX + 1], buf_dst[3 * kX + 2]);
#endif
      break;
    case ExtraTF::kSRGB:
      SIMD_FULL(float) df;
      for (size_t i = 0; i < buf_dst_.xsize(); i += df.N) {
        const auto val = load(df, buf_dst + i);
        const auto result = TF_SRGB().EncodedFromDisplay(val);
        store(result, df, buf_dst + i);
      }
      break;
  }
}

}  // namespace pik
