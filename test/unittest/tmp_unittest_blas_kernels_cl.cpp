#include <stdint.h>
#include <cstring>
#include <assert.h>
#include <vector>
#include <stddef.h>
#include <algorithm>
#include <cmath>

#define QK4_0 32
#define Q4_0 32
#define Q8_0 32

uint32_t ceilDiv(uint32_t a, uint32_t b) { return (a + b - 1) / b; };

float fp32_from_bits(uint32_t w) {
#if defined(__OPENCL_VERSION__)
  return as_float(w);
#elif defined(__CUDA_ARCH__)
  return __uint_as_float((unsigned int)w);
#elif defined(__INTEL_COMPILER)
  return _castu32_f32(w);
#elif defined(_MSC_VER) && (defined(_M_ARM) || defined(_M_ARM64))
  return _CopyFloatFromInt32((__int32)w);
#else
  union {
    uint32_t as_bits;
    float as_value;
  } fp32 = {w};
  return fp32.as_value;
#endif
}

uint32_t fp32_to_bits(float f) {
#if defined(__OPENCL_VERSION__)
  return as_uint(f);
#elif defined(__CUDA_ARCH__)
  return (uint32_t)__float_as_uint(f);
#elif defined(__INTEL_COMPILER)
  return _castf32_u32(f);
#elif defined(_MSC_VER) && (defined(_M_ARM) || defined(_M_ARM64))
  return (uint32_t)_CopyInt32FromFloat(f);
#else
  union {
    float as_value;
    uint32_t as_bits;
  } fp32 = {f};
  return fp32.as_bits;
#endif
}

float compute_fp16_to_fp32(uint16_t h) {
  const uint32_t w = (uint32_t)h << 16;
  const uint32_t sign = w & UINT32_C(0x80000000);
  const uint32_t two_w = w + w;
  const uint32_t exp_offset = UINT32_C(0xE0) << 23;
#if defined(__STDC_VERSION__) && (__STDC_VERSION__ >= 199901L) || \
  defined(__GNUC__) && !defined(__STRICT_ANSI__)
  const float exp_scale = 0x1.0p-112f;
#else
  const float exp_scale = fp32_from_bits(UINT32_C(0x7800000));
#endif
  const float normalized_value =
    fp32_from_bits((two_w >> 4) + exp_offset) * exp_scale;

  const uint32_t magic_mask = UINT32_C(126) << 23;
  const float magic_bias = 0.5f;
  const float denormalized_value =
    fp32_from_bits((two_w >> 17) | magic_mask) - magic_bias;

  const uint32_t denormalized_cutoff = UINT32_C(1) << 27;
  const uint32_t result =
    sign | (two_w < denormalized_cutoff ? fp32_to_bits(denormalized_value)
                                        : fp32_to_bits(normalized_value));
  return fp32_from_bits(result);
}


/**
 * @brief struct template for q4_0 and q8_0
 *
 * @tparam K 4 or 8
 * @return constexpr int number of elements in the quantized block
 */
template <int K> constexpr int QK_0() {
  if constexpr (K == 4) {
    return Q4_0;
  }
  if constexpr (K == 8) {
    return Q8_0;
  }
  return -1;
}

typedef uint16_t nntr_half;

/**
 * @brief block_q4_0
 */
typedef struct {
  nntr_half d;           // delta
  uint8_t qs[QK4_0 / 2]; // nibbles / quants
} block_q4_0;

/**
 * @brief block of q4_0 or q8_0 block
 *
 * @tparam K 4 or 8
 * @tparam N number of blocks to be packed
 */
template <int K, int N> struct block {
  nntr_half d[N];                     // deltas for N qK_0 blocks
  int8_t qs[(QK_0<K>() * N * K) / 8]; // quants for N qK_0 blocks
};

using block_q4_0x8 = block<4, 8>;

static block_q4_0x8 nntr_make_block_q4_0x8(block_q4_0 *in,
                                           unsigned int blck_size_interleave) {
  block_q4_0x8 out;

  for (int i = 0; i < 8; i++) {
    out.d[i] = in[i].d;
  }

  const int end = QK_0<4>() * 4 / blck_size_interleave;
  const uint64_t xor_mask = 0x8888888888888888ULL;

  for (int i = 0; i < end; ++i) {
    int src_id = i % 8;
    int src_offset = (i / 8) * blck_size_interleave;
    int dst_offset = i * blck_size_interleave;

    uint64_t elems;
    memcpy(&elems, &in[src_id].qs[src_offset], sizeof(uint64_t));
    elems ^= xor_mask;
    memcpy(&out.qs[dst_offset], &elems, sizeof(uint64_t));
  }

  return out;
}


int nntr_repack_q4_0_to_q4_0_8_bl(void *__restrict dst, int interleave_block,
                                  const void *__restrict data, size_t data_size,
                                  size_t nrow, size_t k) {
  assert(interleave_block == 8);
  constexpr size_t nrows_interleaved = 8;

  block_q4_0x8 *dst_ = (block_q4_0x8 *)dst;
  const block_q4_0 *src = (const block_q4_0 *)data;
  block_q4_0 dst_tmp[8];
  int nblocks = k / QK_0<4>();

  assert(data_size == nrow * nblocks * sizeof(block_q4_0));

  if (nrow % nrows_interleaved != 0 || k % 8 != 0) {
    return -1;
  }

  for (size_t b = 0; b < nrow; b += nrows_interleaved) {
    for (int64_t x = 0; x < nblocks; x++) {
      for (size_t i = 0; i < nrows_interleaved; i++) {
        dst_tmp[i] = src[x + i * nblocks];
      }
      *dst_++ = nntr_make_block_q4_0x8(dst_tmp, interleave_block);
    }
    src += nrows_interleaved * nblocks;
  }
  return 0;
}
  /// @brief Block size used in the osv32_isv2 layout
  static constexpr const size_t ROW_BLOCK_SIZE = 32;

  /// @brief Numbers of element in one byte of date in the osv32_isv2 layout
  static constexpr const size_t COLUMN_BLOCK_SIZE = 2;

  /**
   * @brief     Compute scale for input weights
   * @param[in] group_weights float * inout vector of weights
   * @param[in] group_size group size (32 or 64 or 128)
   * @return computed scale
   */
float Int4Utils::computeScaleForGroup(const float *group_weights,
                                      const size_t group_size) {
  auto max_absolute_weight = 0.0f;

  for (size_t i = 0; i < group_size; ++i) {
    auto weight = group_weights[i];

    NNTR_THROW_IF(!std::isfinite(weight), std::invalid_argument)
      << "Weight is not finite value";

    const auto absolute_weight = std::abs(weight);

    if (absolute_weight > max_absolute_weight) {
      max_absolute_weight = absolute_weight;
    }
  }

  auto group_scale =
    (max_absolute_weight == 0.0f) ? 1.0f : (max_absolute_weight / 7.0f);

  NNTR_THROW_IF(!std::isfinite(group_scale), std::invalid_argument)
    << "Scale is not finite value";

  return group_scale;
}
  /**
   * @brief     Compute scales for float* matrix weghts
   * @param[in] weights float * input matrix
   * @param[in] rows_count number of rows of input matrix
   * @param[in] columns_count number of columns of input matrix
   * @param[in] group_size group size (32 or 64 or 128)
   * @param[out] scales float vector output scales
   */
void Int4Utils::computeScales(const float *weights, const size_t rows_count,
                              const size_t columns_count,
                              const size_t group_size,
                              std::vector<float> &scales) {
  // NNTR_THROW_IF(columns_count % group_size, std::invalid_argument)
  //   << "Columns size not divisible by group size";
  NNTR_THROW_IF(columns_count % 4, std::invalid_argument)
    << "Columns size not divisible by 4";

  const auto full_groups_per_row = columns_count / group_size;
  const auto last_group_size = columns_count % group_size;
  const auto padded_groups_per_row = ceilDiv(columns_count, group_size);
  const auto rows_count_pad = align(rows_count, ROW_BLOCK_SIZE);
  scales.resize(rows_count_pad * padded_groups_per_row, 1.0f);

  for (size_t row_id = 0; row_id < rows_count; ++row_id) {
    const auto *weights_row = weights + (row_id * columns_count);

    for (size_t group_id = 0; group_id < full_groups_per_row; ++group_id) {
      const auto *weights_group = weights_row + (group_id * group_size);
      scales[(group_id * rows_count_pad) + row_id] =
        computeScaleForGroup(weights_group, group_size);
    }

    // Compute scale for the last padded group
    if (last_group_size > 0) {
      const auto *weights_group =
        weights_row + (full_groups_per_row * group_size);
      scales[(full_groups_per_row * rows_count_pad) + row_id] =
        computeScaleForGroup(weights_group, last_group_size);
    }
  }
}
  /**
   * @brief     Pack one weight from position (row_id, column_id) into 4-bits
   * value
   * @param[in] weights float * input matrix
   * @param[in] scales float * input vector os scales
   * @param[in] row_id number of row
   * @param[in] column_id number of column
   * @param[in] groups_per_row number of groups pre row
   * @param[in] group_size group size (32 or 64 or 128)
   * @param[in] rows_count number of rows of input matrix
   * @param[in] columns_count number of columns of input matrix
   * @return
   */
uint8_t Int4Utils::pack(const float *weights, const float *scales,
                        const size_t row_id, const size_t column_id,
                        const size_t groups_per_row, const size_t group_size,
                        const size_t rows_count, const size_t columns_count) {
  {
    const auto rows_count_pad = align(rows_count, ROW_BLOCK_SIZE);
    const float scale =
      scales[row_id + ((column_id / group_size) * rows_count_pad)];
    const float weight = weights[(row_id * columns_count) + column_id];
    return quantizeToInt4(weight, scale);
  }
}
  /**
   * @brief Quantize weights float* matrix to OpenVINO layout:
   * OS_IS_YX_OSV32_ISV2, osv32_isv2 layout for int4 packed weight:
   *
   * y0_x0x1 | y1_x0x1 | ....  | y15_x0x1|| y16_x0x1 | y17_x0x1 | ... | y31_x0x1
   * y0_x2x3 | y1_x2x3 | ....  | y15_x2x3|| y16_x2x3 | y17_x2x3 | ... | y31_x2x3
   * ...
   * @param weights float * input matrix
   * @param rows_count number of rows of input matrix
   * @param columns_count number of columns of input matrix
   * @param group_size group size (32 or 64 or 128)
   * @param out_weights output quantized weights in layout osv**_isv2
   * @param out_scales output scales
   */
void Int4Utils::quantizeAndRepack(const float *weights, const size_t rows_count,
                                  const size_t columns_count,
                                  const size_t group_size,
                                  std::vector<uint8_t> &out_weights,
                                  std::vector<uint16_t> &out_scales) {
  NNTR_THROW_IF(!weights, std::invalid_argument) << "Weight cannot be null";

  NNTR_THROW_IF((rows_count <= 0), std::invalid_argument)
    << "Rows count needs to be greater than 0";

  NNTR_THROW_IF((columns_count <= 0), std::invalid_argument)
    << "Columns count needs to be greater than 0";

  NNTR_THROW_IF((!(group_size == 32 || group_size == 64 || group_size == 128)),
                std::invalid_argument)
    << "Group size must be 32/64/128";

  std::vector<float> scales_fp32;
  computeScales(weights, rows_count, columns_count, group_size, scales_fp32);

  out_scales.resize(scales_fp32.size());
  for (size_t scale_id = 0; scale_id < scales_fp32.size(); ++scale_id) {
    out_scales[scale_id] = compute_fp32_to_fp16(scales_fp32[scale_id]);
  }

  NNTR_THROW_IF(columns_count % COLUMN_BLOCK_SIZE, std::invalid_argument)
    << "Columns size not divisible by column block size";

  // Prepare output buffer in OS_IS_YX_OSV32_ISV2 layout
  const auto groups_per_row = ceilDiv(columns_count, group_size);
  const auto row_blocks_count = ceilDiv(rows_count, ROW_BLOCK_SIZE);
  const auto columns_count_pad = align(columns_count, group_size);
  const auto column_blocks_count =
    ceilDiv(columns_count_pad, COLUMN_BLOCK_SIZE);
  const auto rows_count_pad = row_blocks_count * ROW_BLOCK_SIZE;

  out_weights.resize((rows_count_pad * columns_count_pad) / 2, 0);

  size_t out_idx = 0;

  for (size_t row_block_id = 0; row_block_id < row_blocks_count;
       ++row_block_id) {
    for (size_t column_block_id = 0; column_block_id < column_blocks_count;
         ++column_block_id) {
      for (size_t i = 0; i < ROW_BLOCK_SIZE; ++i) {
        uint8_t lo = 0, hi = 0;
        const auto row_id_absolute = (row_block_id * ROW_BLOCK_SIZE) + i;
        if (row_id_absolute < rows_count) {
          const auto column_id_absolute_lo =
            (column_block_id * COLUMN_BLOCK_SIZE);
          if (column_id_absolute_lo < columns_count) {
            lo = pack(weights, scales_fp32.data(), row_id_absolute,
                      column_id_absolute_lo, groups_per_row, group_size,
                      rows_count, columns_count);

            const auto column_id_absolute_hi = column_id_absolute_lo + 1;
            if (column_id_absolute_hi < columns_count) {
              hi = pack(weights, scales_fp32.data(), row_id_absolute,
                        column_id_absolute_hi, groups_per_row, group_size,
                        rows_count, columns_count);
            }
          }
        }

        out_weights[out_idx++] = uint8_t((hi << 4) | lo);
      }
    }
  }
}

  /**
   * @brief     Quantize one float value to 4-bits integer
   * @param[in] weight input weight
   * @param[in] scale input scale
   * @return 4-bit integer
   */
uint8_t Int4Utils::quantizeToInt4(const float weight, const float scale) {
  auto div = std::nearbyintf(weight / scale);

  if (std::isnan(div)) {
    div = 0.0f;
  }

  div = std::clamp(div, -8.0f, 7.0f);
  int quantized = (int)div;
  return uint8_t(quantized & 0xF);
}

    /*
        MISSION : VERIFICATION OF TARGET IMPL

        By using Int4Utils::quantizeAndRepack() function, you can obtain osv32_isv2-quantized data from fp32 matrix.
        On the other hand, by using nntrainer::quantize_q4_0 function, you can obtain block_q4_0-quantized data from fp32 matrix. And you should call nntrainer::repack_q4_0() function to get block_q4_0x8-quantized format data by reordering block_q4_0 data.
        Both of the quantization format share similarities like : quantization bit (4bit quantization), and the same number of scale factors, but they are different in data storage order.
        In osv32_isv2, they store 4bit data in their defined way, and store 16bit scale factor explicitly.
        In block_q4_0x8 format, they store 4bit data in re-ordered direction, and store scale factors in 16bit in packed way in specifically defined struct.
        Keeping in these differences between two algorithms in mind,analyze the given code carefully, fully understand what it does, and explain it to me in detail.

        Current mission is to implement transform_q4_0x8_osv32_isv2 function, which transforms the quantized data in osv32_isv2 format into q4_0x8 format by reordering.
    */

  if (K % Q4_0 == 0 && N % 8 == 0) {

  /*
    1. transform osv32_isv2 to q4_0x8 format

    template <int K, int N> struct block {
        nntr_half d[N];                     // deltas for N qK_0 blocks
        int8_t qs[(QK_0<K>() * N * K) / 8]; // quants for N qK_0 blocks
    };
    using block_q4_0x8 = block<4, 8>;

    typedef struct {
        nntr_half d;           // delta
        uint8_t qs[QK4_0 / 2]; // nibbles / quants
    } block_q4_0;

  */
    size_t q4_data_size = K * N / Q4_0 * sizeof(block_q4_0);
    std::vector<float> q4_output_fp32_v2(M * N);
    std::vector<uint8_t> q4_0x8_weight_transformed_from_osv32_isv2(q4_data_size);
    /*
        TARGET IMPLEMENTATION FUNCTION : transform_q4_0x8_osv32_isv2

        Useful codes / functions to take a look:
        - nntr_ggml_impl_common.h
            - struct block_q4_0
            - using block_q4_0x8 = block<4, 8>;
        - nntr_ggml_impl.cpp
            - nntr_repack_q4_0_to_q4_0_8_bl()
        - nntr_ggml_impl_quant.cpp
            - nntr_quantize_q4_0()
        - int4_utils.h
        - int4_utils.cpp
            - quantizeAndRepack
            - pack
            - computeScales
            - quantizeToInt4
        - unittest_blas_kernels_cl.cpp
            - run_int4_gemm_test_

        Algorithm Overview (Most desired)
        [ Method A ]
         (1) From osv32_isv2 quantized_weights.data() and quantized_scales.data(), reorder quantized data into block_q4_0x8 format
         (2) Be careful about the order of quantization parameter in the packed block, by referring  nntr_repack_q4_0_to_q4_0_8_bl and nntr_quantize_q4_0 in nntr_ggml_impl.cpp and nntr_ggml_impl_quant.cpp

        [ Method B ] (Least desired)
         (1) From osv32_isv2 quantized_weights.data() and quantized_scales.data(), reorder quantized data into block_q4_0
         (2) Pack q4_0 to q4_0x8 with SIMD, by referring nntr_repack_q4_0_to_q4_0_8_bl in nntr_ggml_impl.cpp

        [ Method C ] (Not preferred, but easiest)
         (1) Dequantize osv32_isv2 to fp32
         (2) Quantize fp32 to q4_0
         (3) Pack q4_0 to q4_0x8

    */
    nntrainer::transform_q4_0x8_osv32_isv2(N, K, quantized_weights.data(), quantized_scales.data(), scale_group_size /*32*/, q4_0x8_weight_transformed_from_osv32_isv2.data());

  /*
    2. Run GEMM with transformed q4_0x8 weight for verification (FUTURE VERIFICATION TASK! NOT THE SCOPE OF CURRENT WORK)
  */
    // nntrainer::gemm_q4_0(M, N, K, input.data(), K, q4_0x8_weight_transformed_from_osv32_isv2.data(), N,
    //                     q4_output_fp32_v2.data(), N);
    // nntrainer::osv32_isv2_gemm(input_ptr, weight_ptr, scale_ptr, output_ptr, M,
    //                         N, K, scale_group_size); // Sourced from openvino
    //   std::vector<float> output_fp32(M * N);
    // for (unsigned int i = 0; i < M * N; ++i) {
    //     output_fp32[i] = compute_fp16_to_fp32(output_ptr[i]);
    // }
    // float mse_q4 = mse<float>(output_fp32.data(), q4_output_fp32_v2.data(), M * N);
    // std::cout << "MSE Q4_0 W/ osv32_isv2 : " << std::setprecision(10) << mse_q4 << std::endl;

  }
