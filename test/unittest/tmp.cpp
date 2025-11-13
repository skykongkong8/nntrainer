#include <cstddef>
#include <cstdint>
#include <vector>
#include <cstring>
#include <cassert>

// Assumed existing from your code base:
// - Q4_0, ROW_BLOCK_SIZE, COLUMN_BLOCK_SIZE
// - typedef uint16_t nntr_half;
// - struct block_q4_0 { nntr_half d; uint8_t qs[QK4_0 / 2]; };
// - using block_q4_0x8 = block<4, 8>;
// - int nntr_repack_q4_0_to_q4_0_8_bl(void *dst, int interleave_block,
// const void *data, size_t data_size,
// size_t nrow, size_t k);

namespace nntrainer {

void transform_q4_0x8_osv32_isv2(
    size_t N,
    size_t K,
    const uint8_t *osv32_weights,
    const uint16_t *osv32_scales,
    size_t scale_group_size,
    void *dst_q4_0x8) {

  // Basic sanity checks: match constraints used elsewhere in the code.
  assert(osv32_weights != nullptr);
  assert(osv32_scales != nullptr);
  assert(dst_q4_0x8 != nullptr);

  // Blocked kernels assume these divisibility constraints.
  assert(K % Q4_0 == 0); // 32-wide q4_0 blocks
  assert(N % 8 == 0); // 8 rows per q4_0x8 pack

  // Supported group sizes: 32 / 64 / 128. All are multiples of 32.
  assert(scale_group_size == 32 ||
         scale_group_size == 64 ||
         scale_group_size == 128);
  assert(scale_group_size % Q4_0 == 0);

  constexpr size_t ROW_BLOCK_SIZE = 32;
  constexpr size_t COLUMN_BLOCK_SIZE = 2;

  // These must match the definitions used in Int4Utils::quantizeAndRepack()
  const size_t rows_count_pad =
      ((N + ROW_BLOCK_SIZE - 1) / ROW_BLOCK_SIZE) * ROW_BLOCK_SIZE;

  const size_t columns_count_pad =
      ((K + scale_group_size - 1) / scale_group_size) * scale_group_size;

  const size_t column_blocks_count = columns_count_pad / COLUMN_BLOCK_SIZE;
  const size_t nblocks = K / Q4_0; // q4_0 blocks per row

  // Temporary buffer of q4_0 blocks in the layout expected by
  // nntr_repack_q4_0_to_q4_0_8_bl (row-major, nblocks per row).
  std::vector<block_q4_0> tmp_q4;
  tmp_q4.resize(N * nblocks);

  for (size_t r = 0; r < N; ++r) {
    const size_t rb = r / ROW_BLOCK_SIZE;
    const size_t ri = r % ROW_BLOCK_SIZE;

    for (size_t j = 0; j < nblocks; ++j) {
      block_q4_0 &blk = tmp_q4[r * nblocks + j];

      // Initialize block quants to zero (for safety, even if fully filled).
      std::memset(blk.qs, 0, sizeof(blk.qs));

      // Compute and assign the scale (delta) for this 32-wide block.
      const size_t c0 = j * Q4_0;
      const size_t group_id = c0 / scale_group_size;
      const size_t scale_index = r + group_id * rows_count_pad;

      blk.d = osv32_scales[scale_index];

      // Fill 32 int4 values from osv32 layout.
      for (size_t p = 0; p < Q4_0; ++p) {
        const size_t c = c0 + p; // absolute column index, guaranteed < K

        // Map (r,c) into osv32 byte.
        const size_t cb = c / COLUMN_BLOCK_SIZE;
        const size_t idx_byte =
            (rb * column_blocks_count + cb) * ROW_BLOCK_SIZE + ri;

        const uint8_t packed = osv32_weights[idx_byte];
        const uint8_t q = (c & 1)
                            ? (uint8_t)((packed >> 4) & 0xF)
                            : (uint8_t)(packed & 0xF);

        // Store into q4_0 block nibble array.
        const size_t q_byte = p / 2;
        const bool hi = (p & 1) != 0;

        if (!hi) {
          blk.qs[q_byte] =
              (uint8_t)((blk.qs[q_byte] & 0xF0) | q);
        } else {
          blk.qs[q_byte] =
              (uint8_t)((blk.qs[q_byte] & 0x0F) | (uint8_t(q) << 4));
        }
      } // p
    } // j
  } // r

  // Repack 8×q4_0 blocks into block_q4_0x8 layout.
  const size_t data_size = tmp_q4.size() * sizeof(block_q4_0);

  int ret = nntr_repack_q4_0_to_q4_0_8_bl(
      dst_q4_0x8,
      /*interleave_block=*/8,
      tmp_q4.data(),
      data_size,
      /*nrow=*/N,
      /*k=*/K);

  // If this fails, constraints (N%8, K%32) are likely violated.
  assert(ret == 0);
}

} // namespace nntrainer
