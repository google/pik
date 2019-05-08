// Copyright 2019 Google LLC
//
// Use of this source code is governed by an MIT-style
// license that can be found in the LICENSE file or at
// https://opensource.org/licenses/MIT.

// @author Alexander Rhatushnyak

#ifndef PIK_LOSSLESS_C3P_H_
#define PIK_LOSSLESS_C3P_H_

#include <cstddef>
#include <cstdint>

namespace pik {

// TODO(deymo): Remove the anonymous namespace from header files. This
// duplicates the (few) symbols in different objects.
namespace {

const size_t kMaxPlaneMethods = 30;

const int PL1 = 0, PL2 = 1, PL3 = 2;

#define Fsc(buf, bufsize) \
  do {                    \
    datas[sp] = buf;      \
    sizes[sp] = bufsize;  \
    ++sp;                 \
  } while (0)

#define FWr(buf, bufsize)                          \
  do {                                             \
    size_t current = bytes->size();                \
    bytes->resize(bytes->size() + bufsize);        \
    memcpy(bytes->data() + current, buf, bufsize); \
  } while (0)

#define FWrByte(b)    \
  do {                \
    uint8_t byte = b; \
    FWr(&byte, 1);    \
  } while (0)

#if 1  // if 0 to disable all color space transforms
#define compress3planes(pixel_t)                                              \
  do {                                                                        \
    const int spt = sizeof(pixel_t);                                          \
    const int maskPt = (1 << (8 * spt)) - 1, centerPt = (maskPt + 1) >> 1;    \
    size_t compressedCapacity = groupSize2plus * spt;                         \
    uint8_t* compressedData2 = &compressedData[groupSize2plus * spt];         \
    uint8_t* compressedData3 = &compressedData[groupSize2plus * spt * 2];     \
    uint8_t* cd4 = &compressedData[groupSize2plus * spt * 3];                 \
    uint8_t* cd5 = &compressedData[groupSize2plus * spt * 4];                 \
    uint8_t* cd6 = &compressedData[groupSize2plus * spt * 5];                 \
    for (size_t groupY = 0; groupY < ysize; groupY += groupSizeY) {           \
      for (size_t groupX = 0; groupX < xsize; groupX += groupSizeX) {         \
        size_t S1, S2, S3, S4, S5, S6, s1, s2, s3, p1, p2, p3, sizes[3];      \
        uint8_t *cd1, *cd2, *cd3, *datas[3];                                  \
        int sp = 0, planeMethod; /* Here we try guessing which one of the */  \
                                 /* 30 PlaneMethods is best, after trying */  \
                                 /* just six color planes. */                 \
        PIK_RETURN_IF_ERROR(cmprs512x512(&img, PL1, PL1, groupY, groupX,      \
                                         compressedCapacity, compressedData,  \
                                         &s1));                               \
        PIK_RETURN_IF_ERROR(cmprs512x512(&img, PL2, PL2, groupY, groupX,      \
                                         compressedCapacity, compressedData2, \
                                         &s2));                               \
        PIK_RETURN_IF_ERROR(cmprs512x512(&img, PL3, PL3, groupY, groupX,      \
                                         compressedCapacity, compressedData3, \
                                         &s3));                               \
                                                                              \
        S1 = s2, p1 = PL2, cd1 = compressedData2, planeMethod = 10;           \
        S2 = s1, p2 = PL1, cd2 = compressedData;                              \
        S3 = s3, p3 = PL3, cd3 = compressedData3;                             \
        if (s1 < s2 * 63 / 64 && s1 < s3) {                                   \
          S1 = s1, p1 = PL1, cd1 = compressedData, planeMethod = 0;           \
          S2 = s2, p2 = PL2, cd2 = compressedData2;                           \
          S3 = s3, p3 = PL3, cd3 = compressedData3;                           \
        } else if (s3 < s2 * 63 / 64 && s3 < s1) {                            \
          S1 = s3, p1 = PL3, cd1 = compressedData3, planeMethod = 20;         \
          S2 = s1, p2 = PL1, cd2 = compressedData;                            \
          S3 = s2, p3 = PL2, cd3 = compressedData2;                           \
        }                                                                     \
        PIK_RETURN_IF_ERROR(cmprs512x512(&img, p2, p1, groupY, groupX,        \
                                         compressedCapacity, cd4,             \
                                         &S4)); /* R-G+0x8000 */              \
        PIK_RETURN_IF_ERROR(cmprs512x512(&img, p3, p1, groupY, groupX,        \
                                         compressedCapacity, cd5,             \
                                         &S5)); /* B-G+0x8000 */              \
        if (p1 == PL1) {                                                      \
          Fsc(cd1, S1);                                                       \
        }                                                                     \
        if (S4 >= S2 && S5 >= S3) {                                           \
          PIK_RETURN_IF_ERROR(cmprs512x512(&img, p2, p3, groupY, groupX,      \
                                           compressedCapacity, cd6,           \
                                           &S6)); /* R-B+0x8000 */            \
          if (S6 >= S2 && S6 >= S3) {                                         \
            Fsc(cd2, S2);                                                     \
          } else if (S3 > S2 && S3 > S6) {                                    \
            Fsc(cd2, S2);                                                     \
          } else {                                                            \
            Fsc(cd6, S6);                                                     \
          }                                                                   \
          if (p1 == PL2) {                                                    \
            Fsc(cd1, S1);                                                     \
          }                                                                   \
          if (S6 >= S2 && S6 >= S3) {                                         \
            Fsc(cd3, S3);                                                     \
          } else if (S3 > S2 && S3 > S6) {                                    \
            Fsc(cd6, S6);                                                     \
            planeMethod += 5;                                                 \
          } else {                                                            \
            Fsc(cd3, S3);                                                     \
            planeMethod += 4;                                                 \
          }                                                                   \
        } else {                                                              \
          size_t yEnd = std::min(groupSizeY, ysize - groupY) + groupY;        \
          size_t xEnd = std::min(groupSizeX, xsize - groupX);                 \
          size_t p2or3 = (S5 < S4 ? p2 : p3);                                 \
          for (size_t y = groupY; y < yEnd; ++y) {                            \
            pixel_t* PIK_RESTRICT row1 = img.PlaneRow(p1, y) + groupX;        \
            pixel_t* PIK_RESTRICT row2 = img.PlaneRow(p2or3, y) + groupX;     \
            for (size_t x = 0; x < xEnd; ++x) {                               \
              uint32_t v1 = row1[x], v2 = (row2[x] + v1 + centerPt) & maskPt; \
              row2[x] = ((v1 + v2) >> 1) - v1 + centerPt;                     \
            }                                                                 \
          }                                                                   \
          if (S5 < S4) {                                                      \
            PIK_RETURN_IF_ERROR(cmprs512x512(&img, p3, p2, groupY, groupX,    \
                                             compressedCapacity, cd6,         \
                                             &S6)); /* B-RpG/2 */             \
            if (S4 < S2) {                                                    \
              Fsc(cd4, S4);                                                   \
            } else {                                                          \
              Fsc(cd2, S2);                                                   \
            }                                                                 \
            if (p1 == PL2) {                                                  \
              Fsc(cd1, S1);                                                   \
            }                                                                 \
            if (S3 <= S5 && S3 <= S6) {                                       \
              Fsc(cd3, S3);                                                   \
              planeMethod += 1;                                               \
            } else if (S5 <= S6) {                                            \
              Fsc(cd5, S5);                                                   \
              planeMethod += (S4 < S2 ? 3 : 2);                               \
            } else {                                                          \
              Fsc(cd6, S6);                                                   \
              planeMethod += (S4 < S2 ? 6 : 8);                               \
            }                                                                 \
          } else {                                                            \
            PIK_RETURN_IF_ERROR(cmprs512x512(&img, p2, p3, groupY, groupX,    \
                                             compressedCapacity, cd6,         \
                                             &S6)); /* R-BpG/2 */             \
            if (S2 <= S4 && S2 <= S6) {                                       \
              Fsc(cd2, S2);                                                   \
              planeMethod += 2;                                               \
            } else if (S4 <= S6) {                                            \
              Fsc(cd4, S4);                                                   \
              planeMethod += (S5 < S3 ? 3 : 1);                               \
            } else {                                                          \
              Fsc(cd6, S6);                                                   \
              planeMethod += (S5 < S3 ? 7 : 9);                               \
            }                                                                 \
            if (p1 == PL2) {                                                  \
              Fsc(cd1, S1);                                                   \
            }                                                                 \
            if (S5 < S3) {                                                    \
              Fsc(cd5, S5);                                                   \
            } else {                                                          \
              Fsc(cd3, S3);                                                   \
            }                                                                 \
          }                                                                   \
        }                                                                     \
        if (p1 == PL3) {                                                      \
          Fsc(cd1, S1);                                                       \
        }                                                                     \
        FWrByte(planeMethod); /* printf("%2d ", planeMethod); */              \
        FWr(datas[0], sizes[0]);                                              \
        FWr(datas[1], sizes[1]);                                              \
        FWr(datas[2], sizes[2]);                                              \
      } /* groupX */                                                          \
    }   /* groupY */                                                          \
  } while (0)
#else
#define compress3planes(pixel_t)                                              \
  do {                                                                        \
    const int spt = sizeof(pixel_t);                                          \
    size_t compressedCapacity = groupSize2plus * spt;                         \
    uint8_t* compressedData2 = &compressedData[groupSize2plus * spt];         \
    uint8_t* compressedData3 = &compressedData[groupSize2plus * spt * 2];     \
    for (size_t groupY = 0; groupY < ysize; groupY += groupSizeY) {           \
      for (size_t groupX = 0; groupX < xsize; groupX += groupSizeX) {         \
        size_t S1, S2, S3, s1, s2, s3, p1, p2, p3, sizes[3];                  \
        uint8_t *cd1, *cd2, *cd3, *datas[3];                                  \
        int sp = 0, planeMethod; /* Here we try guessing which one of the */  \
                                 /* 30 PlaneMethods is best, after trying */  \
                                 /* just six color planes. */                 \
        PIK_RETURN_IF_ERROR(cmprs512x512(&img, PL1, PL1, groupY, groupX,      \
                                         compressedCapacity, compressedData,  \
                                         &s1));                               \
        PIK_RETURN_IF_ERROR(cmprs512x512(&img, PL2, PL2, groupY, groupX,      \
                                         compressedCapacity, compressedData2, \
                                         &s2));                               \
        PIK_RETURN_IF_ERROR(cmprs512x512(&img, PL3, PL3, groupY, groupX,      \
                                         compressedCapacity, compressedData3, \
                                         &s3));                               \
                                                                              \
        S1 = s1, p1 = PL1, cd1 = compressedData, planeMethod = 0;             \
        S2 = s2, p2 = PL2, cd2 = compressedData2;                             \
        S3 = s3, p3 = PL3, cd3 = compressedData3;                             \
        Fsc(cd1, S1);                                                         \
        Fsc(cd2, S2);                                                         \
        Fsc(cd3, S3);                                                         \
        FWrByte(planeMethod); /* printf("%2d ", planeMethod); */              \
        FWr(datas[0], sizes[0]);                                              \
        FWr(datas[1], sizes[1]);                                              \
        FWr(datas[2], sizes[2]);                                              \
      } /* groupX */                                                          \
    }   /* groupY */                                                          \
  } while (0)
#endif

enum PlaneMethods_30 {  // 8/30 are redundant (left for encoder's convenience)
  RR_G_B = 0,           // p1=R  p2=G  p3=B
  RR_GmR_B = 1,         // p2-p1  p3
  RR_G_BmR = 2,         //   p2  p3-p1
  RR_GmR_BmR = 3,       // p2-p1 p3-p1

  RR_GmB_B = 4,  // == 22   p2-p3 @ p2
  RR_G_GmB = 5,  // ~= 12   p2-p3 @ p3

  RR_GmR_Bm2 = 6,  //  p2-p1  p3-(p1+p2)/2
  RR_Gm2_BmR = 7,  // p2-(p1+p3)/2   p3-p1
  RR_G_Bm2 = 8,    //   p2    p3-(p1+p2)/2
  RR_Gm2_B = 9,    // p2-(p1+p3)/2     p3

  R_GG_B = 10,  // p1=G  p2=R  p3=B
  RmG_GG_B = 11,
  R_GG_BmG = 12,
  RmG_GG_BmG = 13,

  RmB_GG_B = 14,  // == 21
  R_GG_RmB = 15,  // ~=  2

  RmG_GG_Bm2 = 16,
  Rm2_GG_BmG = 17,
  R_GG_Bm2 = 18,
  Rm2_GG_B = 19,

  R_G_BB = 20,  // p1=B  p2=R  p3=G
  RmB_G_BB = 21,
  R_GmB_BB = 22,
  RmB_GmB_BB = 23,

  RmG_G_BB = 24,  // == 11
  R_RmG_BB = 25,  // ~=  1

  RmB_Gm2_BB = 26,
  Rm2_GmB_BB = 27,
  R_Gm2_BB = 28,
  Rm2_G_BB = 29,
};

const uint8_t ncMap[kMaxPlaneMethods] = {
    1 + 2 + 4,  //
    1 + 0 + 4,  //
    1 + 2 + 0,  //
    1,          //
    1 + 0 + 4,  //
    1 + 2 + 0,  //
    1,          //
    1,          //
    1 + 2 + 0,  //
    1 + 0 + 4,  //

    1 + 2 + 4,  //
    0 + 2 + 4,  //
    1 + 2 + 0,  //
    0 + 2 + 0,  //
    0 + 2 + 4,  //
    1 + 2 + 0,  //
    0 + 2 + 0,  //
    0 + 2 + 0,  //
    1 + 2 + 0,  //
    0 + 2 + 4,  //

    1 + 2 + 4,  //
    0 + 2 + 4,  //
    1 + 0 + 4,  //
    0 + 0 + 4,  //
    0 + 2 + 4,  //
    1 + 0 + 4,  //
    0 + 0 + 4,  //
    0 + 0 + 4,  //
    1 + 0 + 4,  //
    0 + 2 + 4,  //
};

#define T3bgn(body)                                  \
  do {                                               \
    for (size_t y = 0; y < yEnd; ++y) {              \
      row1 = img.PlaneRow(PL1, groupY + y) + groupX; \
      row2 = img.PlaneRow(PL2, groupY + y) + groupX; \
      row3 = img.PlaneRow(PL3, groupY + y) + groupX; \
      for (size_t x = 0; x < xEnd; ++x) {            \
        int R = row1[x], G = row2[x], B = row3[x];   \
        (void)R;                                     \
        (void)G;                                     \
        (void)B;                                     \
        { body }                                     \
      }                                              \
    }                                                \
  } while (0)

#define decompress3planes(pixel_t, string8or16)                             \
  do {                                                                      \
    const int spt = sizeof(pixel_t);                                        \
    const int maskPt = (1 << (8 * spt)) - 1, centerPt = (maskPt + 1) >> 1;  \
    for (size_t groupY = 0; groupY < ysize; groupY += groupSizeY) {         \
      for (size_t groupX = 0; groupX < xsize; groupX += groupSizeX) {       \
        if (pos >= compressedSize)                                          \
          return PIK_FAILURE(string8or16 ": out of bounds");                \
        planeMethod = compressedData[pos++];                                \
        if (planeMethod >= kMaxPlaneMethods)                                \
          return PIK_FAILURE(string8or16 ": invalid planeMethod value");    \
        PIK_RETURN_IF_ERROR(dcmprs512x512(&img, PL1, &pos, groupY, groupX,  \
                                          compressedData, compressedSize)); \
        PIK_RETURN_IF_ERROR(dcmprs512x512(&img, PL2, &pos, groupY, groupX,  \
                                          compressedData, compressedSize)); \
        PIK_RETURN_IF_ERROR(dcmprs512x512(&img, PL3, &pos, groupY, groupX,  \
                                          compressedData, compressedSize)); \
                                                                            \
        pixel_t *PIK_RESTRICT row1, *PIK_RESTRICT row2, *PIK_RESTRICT row3; \
        size_t yEnd = std::min(groupSizeY, ysize - groupY);                 \
        size_t xEnd = std::min(groupSizeX, xsize - groupX);                 \
                                                                            \
        switch (planeMethod) {                                              \
          case 0:                                                           \
          case 10:                                                          \
          case 20:                                                          \
            break;                                                          \
          case 1:                                                           \
            T3bgn({                                                         \
              G += R + centerPt;                                            \
              row2[x] = G;                                                  \
            });                                                             \
            break;                                                          \
          case 2:                                                           \
            T3bgn({                                                         \
              B += R + centerPt;                                            \
              row3[x] = B;                                                  \
            });                                                             \
            break;                                                          \
          case 3:                                                           \
            T3bgn({                                                         \
              G += R + centerPt;                                            \
              B += R + centerPt;                                            \
              row2[x] = G;                                                  \
              row3[x] = B;                                                  \
            });                                                             \
            break;                                                          \
          case 22:                                                          \
          case 4:                                                           \
            T3bgn({ row2[x] = G + B + centerPt; });                         \
            break;                                                          \
          case 5:                                                           \
            T3bgn({ row3[x] = G - B + centerPt; });                         \
            break;                                                          \
          case 6:                                                           \
            T3bgn({                                                         \
              row2[x] = G = (G + R + centerPt) & maskPt;                    \
              row3[x] = B + ((R + G) >> 1) + centerPt;                      \
            });                                                             \
            break;                                                          \
          case 7:                                                           \
            T3bgn({                                                         \
              row3[x] = B = (B + R + centerPt) & maskPt;                    \
              row2[x] = G + ((R + B) >> 1) + centerPt;                      \
            });                                                             \
            break;                                                          \
          case 8:                                                           \
            T3bgn({ row3[x] = B + ((R + G) >> 1) + centerPt; });            \
            break;                                                          \
          case 9:                                                           \
            T3bgn({ row2[x] = G + ((R + B) >> 1) + centerPt; });            \
            break;                                                          \
          case 24:                                                          \
          case 11:                                                          \
            T3bgn({                                                         \
              R += G + centerPt;                                            \
              row1[x] = R;                                                  \
            });                                                             \
            break;                                                          \
          case 12:                                                          \
            T3bgn({                                                         \
              B += G + centerPt;                                            \
              row3[x] = B;                                                  \
            });                                                             \
            break;                                                          \
          case 13:                                                          \
            T3bgn({                                                         \
              R += G + centerPt;                                            \
              B += G + centerPt;                                            \
              row1[x] = R;                                                  \
              row3[x] = B;                                                  \
            });                                                             \
            break;                                                          \
          case 21:                                                          \
          case 14:                                                          \
            T3bgn({ row1[x] = R + B + centerPt; });                         \
            break;                                                          \
          case 15:                                                          \
            T3bgn({ row3[x] = R - B + centerPt; });                         \
            break;                                                          \
          case 16:                                                          \
            T3bgn({                                                         \
              row1[x] = R = (R + G + centerPt) & maskPt;                    \
              row3[x] = B + ((R + G) >> 1) + centerPt;                      \
            });                                                             \
            break;                                                          \
          case 17:                                                          \
            T3bgn({                                                         \
              row3[x] = B = (B + G + centerPt) & maskPt;                    \
              row1[x] = R + ((B + G) >> 1) + centerPt;                      \
            });                                                             \
            break;                                                          \
          case 18:                                                          \
            T3bgn({ row3[x] = B + ((R + G) >> 1) + centerPt; });            \
            break;                                                          \
          case 19:                                                          \
            T3bgn({ row1[x] = R + ((B + G) >> 1) + centerPt; });            \
            break;                                                          \
          case 23:                                                          \
            T3bgn({                                                         \
              G += B + centerPt;                                            \
              R += B + centerPt;                                            \
              row1[x] = R;                                                  \
              row2[x] = G;                                                  \
            });                                                             \
            break;                                                          \
          case 25:                                                          \
            T3bgn({ row2[x] = R - G + centerPt; });                         \
            break;                                                          \
          case 26:                                                          \
            T3bgn({                                                         \
              row1[x] = R = (R + B + centerPt) & maskPt;                    \
              row2[x] = G + ((B + R) >> 1) + centerPt;                      \
            });                                                             \
            break;                                                          \
          case 27:                                                          \
            T3bgn({                                                         \
              row2[x] = G = (G + B + centerPt) & maskPt;                    \
              row1[x] = R + ((B + G) >> 1) + centerPt;                      \
            });                                                             \
            break;                                                          \
          case 28:                                                          \
            T3bgn({ row2[x] = G + ((B + R) >> 1) + centerPt; });            \
            break;                                                          \
          case 29:                                                          \
            T3bgn({ row1[x] = R + ((B + G) >> 1) + centerPt; });            \
            break;                                                          \
          default:                                                          \
            return PIK_FAILURE(string8or16 ": invalid planeMethod value");  \
        }                                                                   \
      } /* groupX */                                                        \
    }   /* groupY */                                                        \
  } while (0)

//#undef T3bgn

}  // namespace
}  // namespace pik

#endif  // PIK_LOSSLESS_C3P_H_
