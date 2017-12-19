#ifndef COMMON_H_
#define COMMON_H_

namespace pik {

inline int DivCeil(int a, int b) { return (a + b - 1) / b; }

constexpr int kTileSize = 64;
constexpr int kTileInBlocks = kTileSize >> 3;
constexpr int kSupertileInBlocks = 64;

}  // namespace pik

#endif  // COMMON_H_
