#ifndef COMMON_H_
#define COMMON_H_

#include <stddef.h>

namespace pik {

inline int DivCeil(int a, int b) { return (a + b - 1) / b; }

/* Constant suffix "InXxx" express measuring unit.
   Empty suffix stands for pixels. */

/* Block is the rectangular grid of pixels to which "energy compaction"
   transformation (e.g. DCT) is applied. */
constexpr size_t kBlockWidth = 8;
constexpr size_t kBlockHeight = kBlockWidth;
/* "size" is the number of coefficients required to describe transformed block.
   Coincidentally, it is also the number of pixels in block. */
constexpr size_t kBlockSize = kBlockWidth * kBlockHeight;

/* Tile is the rectangular grid of blocks
   that share color transform parameters. */
constexpr size_t kTileWidthInBlocks = 8;
constexpr size_t kTileHeightInBlocks = kTileWidthInBlocks;
constexpr size_t kTileWidth = kTileWidthInBlocks * kBlockWidth;
constexpr size_t kTileHeight = kTileHeightInBlocks * kBlockHeight;

/* Group is the rectangular grid of tiles that could be decoded independently.
   Multiple groups could be decoded in parallel. */
constexpr size_t kGroupWidthInTiles = 8;
constexpr size_t kGroupHeightInTiles = kGroupWidthInTiles;
constexpr size_t kGroupWidthInBlocks = kGroupWidthInTiles * kTileWidthInBlocks;
constexpr size_t kGroupHeightInBlocks =
    kGroupHeightInTiles * kTileHeightInBlocks;

}  // namespace pik

#endif  // COMMON_H_
