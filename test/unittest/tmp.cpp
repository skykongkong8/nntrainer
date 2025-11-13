#include <cstddef>
#include <cstdint>
#include <vector>
#include <cstring>
#include <cassert>

namespace nntrainer {

// From nntr_ggml_impl_common.h:
//
// #define Q4_0 32
// struct block_q4_0 {
// nntr_half d;
// uint8_t qs[Q4_0 / 2];
// };
//
// using block_q4_0x8 = block<4, 8>;
//
// int nntr_repack_q4_0_to_q4_0_8_bl(void *__restrict dst,
// int interleave_block,
// const void *__restrict data,
// size_t data_size,
// size_t nrow,
// size_t k);

void transform_q4_0x8_osv32_isv2(
    size_t N,
    size_t K,
    const uint8_t *osv32_weights,
    const uint16_t *osv32_scales,
    size_t scale_group_size,
    void *dst_q4_0x8)
{
  assert(osv32_weights);
  assert(osv32_scales);
  assert(dst_q4_0x8);

  // q4_0 + q4_0x8 structural constraints
  assert(K % Q4_0 == 0); // 32-wide q4_0 blocks along K
  assert(N % 8 == 0); // rows multiple of 8 for q4_0x8

  assert(scale_group_size == 32 ||
         scale_group_size == 64 ||
         scale_group_size == 128);
  assert(scale_group_size % Q4_0 == 0);

  constexpr size_t ROW_BLOCK_SIZE = 32; // osv32 row tile
  constexpr size_t COLUMN_BLOCK_SIZE = 2; // 2 columns per byte

  // osv32 padding (must mirror Int4Utils::quantizeAndRepack)
  const size_t row_blocks_count =
      (N + ROW_BLOCK_SIZE - 1) / ROW_BLOCK_SIZE;
  const size_t rows_count_pad =
      row_blocks_count * ROW_BLOCK_SIZE;

  const size_t columns_count_pad =
      ((K + scale_group_size - 1) / scale_group_size) * scale_group_size;
  const size_t column_blocks_count =
      columns_count_pad / COLUMN_BLOCK_SIZE;

  const size_t nblocks = K / Q4_0; // q4_0 blocks per row

  // Temporary q4_0 buffer laid out as [row, block_index]
  std::vector<block_q4_0> q4_blocks(N * nblocks);

  for (size_t r = 0; r < N; ++r) {
    const size_t rb = r / ROW_BLOCK_SIZE; // osv32 row-block index
    const size_t ri = r % ROW_BLOCK_SIZE; // row inside 32-row block

    // Base byte index in osv32 for this row and column_block=0
    const size_t row_block_base =
        (rb * column_blocks_count) * ROW_BLOCK_SIZE + ri;

    for (size_t j = 0; j < nblocks; ++j) {
      block_q4_0 &blk = q4_blocks[r * nblocks + j];

      // Clear q4_0 bytes to allow nibble RMW
      std::memset(blk.qs, 0, sizeof(blk.qs));

      const size_t c0 = j * Q4_0; // first column of this block
      const size_t group_id = c0 / scale_group_size;
      const size_t scale_index = r + group_id * rows_count_pad;

      // Use osv32-provided scale for this row & column group
      blk.d = osv32_scales[scale_index];

      // Fill 32 values in this q4_0 block
      for (size_t p = 0; p < Q4_0; ++p) {
        const size_t c = c0 + p; // absolute column index

        const size_t cb = c / COLUMN_BLOCK_SIZE; // osv32 column-block index
        const size_t idx_byte =
            row_block_base + cb * ROW_BLOCK_SIZE; // byte index in osv32

        const uint8_t packed = osv32_weights[idx_byte];

        // Extract the nibble for (r, c) as-is
        uint8_t nib;
        if (c & 1) {
          nib = (uint8_t)((packed >> 4) & 0xF);
        } else {
          nib = (uint8_t)(packed & 0xF);
        }

        // Place nib into q4_0's (byte_idx, low/high) position
        const size_t byte_idx = p / 2;
        const bool hi = (p & 1) != 0;

        if (!hi) {
          blk.qs[byte_idx] =
              (uint8_t)((blk.qs[byte_idx] & 0xF0u) | nib);
        } else {
          blk.qs[byte_idx] =
              (uint8_t)((blk.qs[byte_idx] & 0x0Fu) | (uint8_t(nib) << 4));
        }
      } // p
    } // j
  } // r

  const size_t data_size = q4_blocks.size() * sizeof(block_q4_0);

  // Canonical repack into q4_0x8
  int ret = nntr_repack_q4_0_to_q4_0_8_bl(
      dst_q4_0x8,
      /*interleave_block=*/8,
      q4_blocks.data(),
      data_size,
      /*nrow=*/N,
      /*k=*/K);

  assert(ret == 0);
}
