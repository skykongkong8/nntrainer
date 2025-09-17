/*++

Copyright (c) Microsoft Corporation. All rights reserved.

Licensed under the MIT License.

Module Name:

    mlas_q4.h

Abstract:

    This module contains the public data structures and procedure prototypes
    for blocked int4 quantization and dequantization.

    Int4 block quantization is used to compress weight tensors of large
    language models.

--*/

#pragma once

#include "mlas.h"
#include "mlas_gemm_postprocessor.h"

#include <math.h>
#include <algorithm>

/**
 * @brief Define types of block quantization
 */
typedef enum {
    BlkQ4Sym = 0,    /*!< int4 Symmetric Block Quantization, zero_point = 0 */
    BlkQ4Zp8 = 1,    /*!< int4 Block Quantization, zero_point is int8 type */
    BlkQ4Sym64 = 2,  /*!< int4 Symmetric Block Quantization, 64 values per block*/
    BlkQ4Sym128 = 4  /*!< int4 Symmetric Block Quantization, 128 values per block*/
} MLAS_BLK_QUANT_TYPE;

/**
 * @brief Computes the number of bytes required to pack and int4-quantize
 *        a weight matrix
 * @param QType  type of block quantization
 * @param N      the number of columns of matrix B.
 * @param K      the number of rows of matrix B.
 * @return size of the packing buffer, 0 if the operation is not yet supported.
*/
size_t
MLASCALL
MlasQ4GemmPackBSize(
    MLAS_BLK_QUANT_TYPE QType,
    size_t N,
    size_t K
    );

/**
 * @brief Prepack and Quantize fp32 weight tensor to int4 blocks
 *
 * @param QType      type of block quantization
 * @param PackedBuf  destination buffer
 * @param FpData     the pointer to fp32 matrix
 * @param N          the number of columns of matrix B.
 * @param K          the number of rows of matrix B.
 * @param ldb        leading dimension of B
*/
void
MLASCALL
MlasQ4GemmPackB(
    MLAS_BLK_QUANT_TYPE QType,
    void* PackedBuf,
    const float* FpData,
    size_t N,
    size_t K,
    size_t ldb
    );


/**
 * @brief Unpack and dequantize from int4 to fp32, reverse operation of
 *        MlasQ4GemmPackB
 * @param QType      type of block quantization
 * @param FpData     destination buffer, the fp32 matrix
 * @param PackedBuf  int4 quantized and packed data
 * @param N          the number of columns of matrix B.
 * @param K          the number of rows of matrix B.
 * @param ldb        leading dimension of B
 */
void
MLASCALL
MlasQ4GemmUnPackB(
    MLAS_BLK_QUANT_TYPE QType,
    float* FpData,
    const void* PackedBuf,
    size_t N,
    size_t K,
    size_t ldb
    );


/**
 * @brief Data parameters for Q4 GEMM routine
 *        C = A * B + Bias
 *        A must be a float32 matrix
 *        B must be a quantized and packed int4 blob
 *        All except C are [in] parameters
 */
struct MLAS_Q4_GEMM_DATA_PARAMS {
    const float* A = nullptr;        /**< address of A (float32 matrix)*/
    const void* B = nullptr;         /**< address of B (quantized and packed int4 blob)*/
    const float* Bias = nullptr;     /**< address of Bias, vector size N */
    float* C = nullptr;              /**< address of result matrix */
    size_t lda = 0;                  /**< leading dimension of A */
    size_t ldc = 0;                  /**< leading dimension of C*/
    const MLAS_GEMM_POSTPROCESSOR<float>* OutputProcessor = nullptr;
};

/**
 * @brief Batched GEMM:  C = A * B + Bias
 *        A must be a float32 matrix
 *        B must be a quantized and packed int4 blob
 *
 * @param[in]  QType   type of block quantization used in B
 * @param[in]  M       row size of matrix A and C
 * @param[in]  N       column size of matrix B and C
 * @param[in]  K       column size of matrix A and row size of matrix B
 * @param[in]  BatchN  number of batches
 * @param[inout]  DataParams  An array (size BatchN) of parameter blocks
 * @param[in]  ThreadPool
 * @return
 */
void MLASCALL
MlasQ4GemmBatch(
    MLAS_BLK_QUANT_TYPE QType,
    const size_t M,
    const size_t N,
    const size_t K,
    const size_t BatchN,
    const MLAS_Q4_GEMM_DATA_PARAMS* DataParams,
    MLAS_THREADPOOL* ThreadPool = nullptr
    );


/**
 * @brief Calculate the buffer size needed for int8 block quantize
 * @param[in]  QType   Type of block quantization used
 * @param[in]  M       Number of rows of the input matrix
 * @param[in]  K       Number of columns of the input matrix
 * @return    buffer size (in bytes) needed, 0 if not yet supported on current hardware
*/
size_t
MLASCALL
MlasQ80BlkQuantSize(MLAS_BLK_QUANT_TYPE QType, size_t M, size_t K);

/**
 * @brief Given an input float 2-D matrix, perform blocked int8 quantize
 *
 * @param QType     Type of block quantization used
 * @param Qblob     Pointer to the output buffer
 * @param A         Pointer to the float matrix
 * @param M         Number of rows of the input matrix
 * @param K         Number of columns of the input matrix
 * @param lda       leading dimension of the input matrix
 * @param ThreadPool
*/
void
MLASCALL
MlasQ80BlkQuant(
    MLAS_BLK_QUANT_TYPE QType,
    void* Qblob,
    const float* A,
    size_t M,
    size_t K,
    size_t lda,
    MLAS_THREADPOOL* ThreadPool
    );


/**
 * @brief Data parameters for Q8Q4 GEMM routine
 *        C = A * B + Bias
 *        A must be a block quantized int8 matrix
 *        B must be a block quantized and packed int4 blob
 *        All except C are [in] parameters
 */
struct MLAS_Q8Q4_GEMM_DATA_PARAMS {
    const void* A = nullptr;     /**< address of A (quantized int8 blob)*/
    const void* B = nullptr;     /**< address of B (quantized and packed int4 blob)*/
    const float* Bias = nullptr; /**< address of Bias, vector size N */
    float* C = nullptr;          /**< address of result matrix */
    size_t ldc = 0;              /**< leading dimension of C*/
    const MLAS_GEMM_POSTPROCESSOR<float>* OutputProcessor = nullptr;
};

/**
 * @brief Batched GEMM:  C = A * B + Bias
 *        A must be a quantized int8 blob
 *        B must be a quantized and packed int4 blob
 *
 * @param[in]  QType   type of block quantization used in B
 * @param[in]  M       row size of matrix A and C
 * @param[in]  N       column size of matrix B and C
 * @param[in]  K       column size of matrix A and row size of matrix B
 * @param[in]  BatchN  number of batches
 * @param[inout]  DataParams  An array (size BatchN) of parameter blocks
 * @param[in]  ThreadPool
 * @return
 */
void MLASCALL
MlasQ8Q4GemmBatch(
    MLAS_BLK_QUANT_TYPE QType,
    const size_t M,
    const size_t N,
    const size_t K,
    const size_t BatchN,
    const MLAS_Q8Q4_GEMM_DATA_PARAMS* DataParams,
    MLAS_THREADPOOL* ThreadPool
    );


////////////////////////////////////////////////////////////
// Blockwise quantization and dequantization where quantization
// parameters are packed into separate buffers.
//

/**
 * @brief For quantization type <T, block_size, columnwise>, and
 *        matrix shape [rows, columns], compute the shape of the
 *        quantization parameter matrix [meta_rows, meta_cols]
*/
template <typename T, int qbits>
void
MlasBlockwiseQuantMetaShape(
    int block_size,
    bool columnwise,
    int rows,
    int columns,
    int& meta_rows,
    int& meta_cols
    );

/**
 * @brief For quantization type <T, block_size, columnwise>, and
 * matrix shape [rows, columns], compute the shape of the
 * quantized matrix [q_rows, q_cols]. The quantized matrix
 * is in column major layout, with bits packed on the column.
 *
 * @tparam T
 * @tparam qbits
 * @param block_size
 * @param columnwise
 * @param rows
 * @param columns
 * @param q_rows
 * @param q_cols
*/
template <typename T, int qbits>
void
MlasBlockwiseQuantizedShape(
    int block_size,
    bool columnwise,
    int rows,
    int columns,
    int& q_rows,
    int& q_cols
    );

/**
 * @brief Compute the sizes of the quantized data and quantization parameter buffers.
 *
 * @param qbits                             The bit width of each quantized value.
 * @param block_size                        The number of quantized values in a block.
 * @param columnwise                        Whether a block contains values from a matrix column (true) or row (false).
 * @param rows                              Number of matrix rows.
 * @param columns                           Number of matrix columns.
 * @param[out] q_data_size_in_bytes         The size in bytes of the quantized data.
 * @param[out] q_scale_num_elements         The size in elements of the scale quantization parameters.
 * @param[out] q_zero_point_size_in_bytes   The size in bytes of the zero point quantization parameters. Optional.
 *
 * If the qbits or block_size values are unsupported the output sizes will be zero.
 */
void MLASCALL
MlasBlockwiseQuantizedBufferSizes(
    int qbits,
    int block_size,
    bool columnwise,
    int rows,
    int columns,
    size_t& q_data_size_in_bytes,
    size_t& q_scale_num_elements,
    size_t* q_zero_point_size_in_bytes
);


/**
 * @brief Blockwise 4 bits quantization, resulting elements and quantization
 *        parameters (scales, zero points) are packed into separate matrices
 *        all in column major layout for faster access during subsequent matrix
 *        multiplication.
 *
 * @tparam ElementT             type of the input matrix element, usually floating point
 * @tparam qbits                number of bits used for quantization, 4 for int4
 *
 * @param dst                   points to the quantized matrix, shape [rows, columns] column major
 * @param scales                points to the scales matrix, column major
 * @param zero_points           points to the zero_points matrix, column major
 * @param src                   points to the floating point matrix, to be quantized, row major shape [rows, columns]
 * @param block_size            size of the block to quantize, elements from the same block share the same scale and zero point
 * @param columnwise            true when elements in a block are from the same column, false when elements in a block are from the same row
 * @param rows
 * @param columns
 * @param leading_dimension
 * @param thread_pool
*/
template <typename ElementT, int qbits>
void
MlasQuantizeBlockwise(
    uint8_t* dst,
    ElementT* scales,
    uint8_t* zero_points,
    const ElementT* src,
    int block_size,
    bool columnwise,
    int rows,
    int columns,
    int leading_dimension,
    MLAS_THREADPOOL* thread_pool
    );


/**
 * @brief Blockwise 4 bits dequantization, quantized elements and quantization
 *        parameters (scales, zero points) are from separate matrices packed
 *        in column major layout.  Output is a floating point matrix in column
 *        major layout for faster access during subsequent matrix multiplication.
 *
 * @tparam ElementT     type of the dequantized matrix element, usually floating point
 * @tparam qbits        number of bits used for quantization, 4 for int4
 *
 * @param dst           points to dequantized matrix shape [rows, columns] column major
 * @param src           points to quantized matrix, column major
 * @param scales        points to quantization scales, column major
 * @param zero_points   points to quantization zero points, column major
 * @param block_size    size of the block to quantize, elements from the same block share the same scale and zero point
 * @param columnwise    true when elements in a block are from the same column, false when elements in a block are from the same row
 * @param rows
 * @param columns
 * @param thread_pool
*/
template <typename ElementT, int qbits>
void
MlasDequantizeBlockwise(
    ElementT* dst,
    const uint8_t* src,
    const ElementT* scales,
    const uint8_t* zero_points,
    int block_size,
    bool columnwise,
    int rows,
    int columns,
    MLAS_THREADPOOL* thread_pool
    );

/**
 * @brief Blockwise 4 bits quantization. After quantization, the weights and zero points
 *        are packed row-wise. If zero_points is null, quantized type is int4 with default
 *        zero point 0, to align with DQ schema. Otherwise, quantized type is uint4.
 *        In int4/uint4, dst have the same shape as src, and zero_points have the same shape as scales.
 * @tparam Tin
 * @tparam qbits            number of bits used for quantization, only 4 is supported
 * @param src               points to the floating point matrix, to be quantized, row major shape [rows, columns]
 * @param scales            points to the scales matrix, row major
 * @param zero_points       points to the zero_points matrix, row major
 * @param dst               points to the quantized matrix, shape [rows, columns] row major in qbits type.
 *                          In uint8_t type, shape is [rows, columns * qbits / 8].
 * @param columnwise        true when quantize elements in a column, false when quantize elements in a row.
 * @param rows
 * @param columns
 * @param quant_block_size  number of elements in a quantize block
 * @param thread_pool
 * @return the quantized type is signed.
 */
template <typename Tin, int qbits>
bool
MlasQDQQuantizeBlockwise(
    const Tin* src,
    Tin* scales,
    uint8_t* zero_points,
    uint8_t* dst,
    bool columnwise,
    int rows,
    int columns,
    int quant_block_size,
    MLAS_THREADPOOL* thread_pool
);

/**
 * @brief Transpose blockwise quantized tensors. The src tensors are row major. src weights and zero
 *        points are packed row-wise. The dst tensors are column major. dst weights and zero points
 *        are packed column-wise.
 *        dst_weights and dst_zero_points are in uint4.
 *        If src_weights is int4 and has src_zero_points, src_weights and src_zero_points are
 *        converted to uint4 by adding 8.
 *        If src_weights is int4 and no src_zero_points, src_weights is converted to uint4 by adding 8.
 *        src_zero_points is 0 and dst_zero_points is 8.
 *        If src_weights is uint4 and has src_zero_points, just transpose.
 *        If src_weights is uint4 and no src_zero_points, caller must allocate dst_zero_points with
 *        0 values. Otherwise exception is thrown.
 * @tparam Tin
 * @tparam qbits            number of bits used for quantization, only 4 is supported
 * @tparam signed_quant     true when quantized type is signed, false when quantized type is unsigned
 * @param src_weights       points to the quantized matrix, row major, shape [rows, columns] in qbits type.
 *                          In uint8_t type, shape is [rows, columns * qbits / 8].
 * @param src_scales        points to the scales matrix, row major
 * @param src_zero_points   points to the zero_points matrix, row major. Packed row-wise.
 * @param dst_weights       points to the quantized matrix, column major. Packed column-wise.
 * @param dst_scales        points to the scales matrix, column major
 * @param dst_zero_points   points to the zero_points matrix, column major. Packed column-wise.
 * @param columnwise        true when quantize elements in a column, false when quantize elements in a row.
 * @param rows
 * @param columns
 * @param quant_block_size  number of elements in a quantize block
 * @param thread_pool
 */
template <typename Tin, int qbits, bool signed_quant>
void
MlasQDQTransposeBlockwiseQuantized(
    const uint8_t* src_weights,
    const Tin* src_scales,
    const uint8_t* src_zero_points,
    uint8_t* dst_weights,
    Tin* dst_scales,
    uint8_t* dst_zero_points,
    bool columnwise,
    int rows,
    int columns,
    int quant_block_size,
    MLAS_THREADPOOL* thread_pool
);

/*
In file included from /usr/include/c++/13/bits/requires_hosted.h:31,
                 from /usr/include/c++/13/system_error:34,
                 from /home/sungsik/nntrainer/nntrainer/tensor/cpu_backend/sqnbitgemm_interface/sqnbitgemm_platform.cpp:18:
/usr/include/c++/13/bits/chrono.h:1228:1: error: expected unqualified-id before ‘namespace’
 1228 | _GLIBCXX_BEGIN_INLINE_ABI_NAMESPACE(_V2)
      | ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
In file included from /usr/include/c++/13/bits/this_thread_sleep.h:36,
                 from /usr/include/c++/13/thread:46,
                 from /home/sungsik/nntrainer/nntrainer/tensor/cpu_backend/sqnbitgemm_interface/sqnbitgemm_platform.cpp:34:
/usr/include/c++/13/bits/chrono.h:1329:10: error: expected unqualified-id before ‘namespace’
 1329 |   inline namespace literals
      |          ^~~~~~~~~
/usr/include/c++/13/bits/chrono.h:1447:31: error: ‘chrono_literals’ is not a namespace-name; did you mean ‘string_literals’?
 1447 |     using namespace literals::chrono_literals;
      |                               ^~~~~~~~~~~~~~~
      |                               string_literals
/usr/include/c++/13/bits/chrono.h:1482:35: error: ‘system_clock’ in namespace ‘std::chrono’ does not name a type
 1482 |       using __sys_clock = chrono::system_clock;
      |                                   ^~~~~~~~~~~~
/usr/include/c++/13/bits/chrono.h:1494:46: error: ‘__sys_clock’ was not declared in this scope; did you mean ‘__file_clock’?
 1494 |         _S_from_sys(const chrono::time_point<__sys_clock, _Dur>& __t) noexcept
      |                                              ^~~~~~~~~~~
      |                                              __file_clock
/usr/include/c++/13/bits/chrono.h:1494:63: error: template argument 1 is invalid
 1494 |         _S_from_sys(const chrono::time_point<__sys_clock, _Dur>& __t) noexcept
      |                                                               ^
/usr/include/c++/13/bits/chrono.h:1503:28: error: ‘__sys_clock’ was not declared in this scope; did you mean ‘__file_clock’?
 1503 |         chrono::time_point<__sys_clock, _Dur>
      |                            ^~~~~~~~~~~
      |                            __file_clock
/usr/include/c++/13/bits/chrono.h:1503:45: error: template argument 1 is invalid
 1503 |         chrono::time_point<__sys_clock, _Dur>
      |                                             ^
/usr/include/c++/13/bits/chrono.h: In static member function ‘static std::filesystem::__file_clock::time_point std::filesystem::__file_clock::now()’:
/usr/include/c++/13/bits/chrono.h:1464:36: error: ‘std::chrono::system_clock’ has not been declared
 1464 |       { return _S_from_sys(chrono::system_clock::now()); }
      |                                    ^~~~~~~~~~~~
/usr/include/c++/13/bits/chrono.h: In static member function ‘static std::chrono::time_point<std::filesystem::__file_clock, _Dur> std::filesystem::__file_clock::_S_from_sys(const int&)’:
/usr/include/c++/13/bits/chrono.h:1497:34: error: request for member ‘time_since_epoch’ in ‘__t’, which is of non-class type ‘const int’
 1497 |           return __file_time{__t.time_since_epoch()} - _S_epoch_diff;
      |                                  ^~~~~~~~~~~~~~~~
/usr/include/c++/13/bits/chrono.h: In static member function ‘static int std::filesystem::__file_clock::_S_to_sys(const std::chrono::time_point<std::filesystem::__file_clock, _Dur>&)’:
/usr/include/c++/13/bits/chrono.h:1506:49: error: ‘__sys_clock’ was not declared in this scope; did you mean ‘__file_clock’?
 1506 |           using __sys_time = chrono::time_point<__sys_clock, _Dur>;
      |                                                 ^~~~~~~~~~~
      |                                                 __file_clock
/usr/include/c++/13/bits/chrono.h:1506:66: error: template argument 1 is invalid
 1506 |           using __sys_time = chrono::time_point<__sys_clock, _Dur>;
      |                                                                  ^
/usr/include/c++/13/bits/chrono.h:1507:18: error: ‘__sys_time’ was not declared in this scope; did you mean ‘SYS_time’?
 1507 |           return __sys_time{__t.time_since_epoch()} + _S_epoch_diff;
      |                  ^~~~~~~~~~
      |                  SYS_time
/usr/include/c++/13/bits/chrono.h:1507:28: error: expected ‘;’ before ‘{’ token
 1507 |           return __sys_time{__t.time_since_epoch()} + _S_epoch_diff;
      |                            ^
/usr/include/c++/13/bits/chrono.h:1507:51: error: expected ‘;’ before ‘}’ token
 1507 |           return __sys_time{__t.time_since_epoch()} + _S_epoch_diff;
      |                                                   ^
In file included from /home/sungsik/nntrainer/nntrainer/tensor/cpu_backend/sqnbitgemm_interface/sqnbitgemm_platform.cpp:35:
/usr/include/c++/13/mutex: At global scope:
/usr/include/c++/13/mutex:172:60: error: ‘system_clock’ is not a member of ‘std::chrono’
  172 |         _M_try_lock_until(const chrono::time_point<chrono::system_clock,
      |                                                            ^~~~~~~~~~~~
/usr/include/c++/13/mutex:173:61: error: template argument 1 is invalid
  173 |                                                    _Duration>& __atime)
      |                                                             ^
/usr/include/c++/13/mutex:189:60: error: ‘steady_clock’ is not a member of ‘std::chrono’
  189 |         _M_try_lock_until(const chrono::time_point<chrono::steady_clock,
      |                                                            ^~~~~~~~~~~~
/usr/include/c++/13/mutex:190:61: error: template argument 1 is invalid
  190 |                                                    _Duration>& __atime)
      |                                                             ^
/usr/include/c++/13/mutex:189:9: error: ‘template<class _Derived> template<class _Duration> bool std::__timed_mutex_impl<_Derived>::_M_try_lock_until(const int&)’ cannot be overloaded with ‘template<class _Derived> template<class _Duration> bool std::__timed_mutex_impl<_Derived>::_M_try_lock_until(const int&)’
  189 |         _M_try_lock_until(const chrono::time_point<chrono::steady_clock,
      |         ^~~~~~~~~~~~~~~~~
/usr/include/c++/13/mutex:172:9: note: previous declaration ‘template<class _Derived> template<class _Duration> bool std::__timed_mutex_impl<_Derived>::_M_try_lock_until(const int&)’
  172 |         _M_try_lock_until(const chrono::time_point<chrono::system_clock,
      |         ^~~~~~~~~~~~~~~~~
/usr/include/c++/13/mutex: In member function ‘bool std::__timed_mutex_impl<_Derived>::_M_try_lock_for(const std::chrono::duration<_Rep, _Period>&)’:
/usr/include/c++/13/mutex:159:35: error: ‘steady_clock’ in namespace ‘std::chrono’ does not name a type
  159 |           using __clock = chrono::steady_clock;
      |                                   ^~~~~~~~~~~~
/usr/include/c++/13/mutex:164:45: error: ‘__clock’ was not declared in this scope; did you mean ‘clock’?
  164 |           auto __rt = chrono::duration_cast<__clock::duration>(__rtime);
      |                                             ^~~~~~~
      |                                             clock
/usr/include/c++/13/mutex:164:31: error: parse error in template argument list
  164 |           auto __rt = chrono::duration_cast<__clock::duration>(__rtime);
      |                               ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
/usr/include/c++/13/mutex:165:53: error: wrong number of template arguments (1, should be 2)
  165 |           if (ratio_greater<__clock::period, _Period>())
      |                                                     ^
In file included from /usr/include/c++/13/bits/chrono.h:37:
/usr/include/c++/13/ratio:415:12: note: provided for ‘template<class _R1, class _R2> struct std::ratio_greater’
  415 |     struct ratio_greater
      |            ^~~~~~~~~~~~~
/usr/include/c++/13/mutex:167:36: error: ‘__clock’ is not a class, namespace, or enumeration
  167 |           return _M_try_lock_until(__clock::now() + __rt);
      |                                    ^~~~~~~
/usr/include/c++/13/mutex: In member function ‘bool std::__timed_mutex_impl<_Derived>::_M_try_lock_until(const int&)’:
/usr/include/c++/13/mutex:175:62: error: no matching function for call to ‘time_point_cast<std::chrono::seconds>(const int&)’
  175 |           auto __s = chrono::time_point_cast<chrono::seconds>(__atime);
      |                      ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~^~~~~~~~~
/usr/include/c++/13/bits/chrono.h:1030:7: note: candidate: ‘template<class _ToDur, class _Clock, class _Dur> constexpr std::__enable_if_t<std::chrono::__is_duration<_Tp>::value, std::chrono::time_point<_Clock, _ToDur> > std::chrono::time_point_cast(const time_point<_Clock, _Dur>&)’
 1030 |       time_point_cast(const time_point<_Clock, _Dur>& __t)
      |       ^~~~~~~~~~~~~~~
/usr/include/c++/13/bits/chrono.h:1030:7: note:   template argument deduction/substitution failed:
/usr/include/c++/13/mutex:175:62: note:   mismatched types ‘const std::chrono::time_point<_Clock, _Dur>’ and ‘const int’
  175 |           auto __s = chrono::time_point_cast<chrono::seconds>(__atime);
      |                      ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~^~~~~~~~~
/usr/include/c++/13/mutex: In member function ‘bool std::__timed_mutex_impl<_Derived>::_M_try_lock_until(const int&)’:
/usr/include/c++/13/mutex:192:62: error: no matching function for call to ‘time_point_cast<std::chrono::seconds>(const int&)’
  192 |           auto __s = chrono::time_point_cast<chrono::seconds>(__atime);
      |                      ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~^~~~~~~~~
/usr/include/c++/13/bits/chrono.h:1030:7: note: candidate: ‘template<class _ToDur, class _Clock, class _Dur> constexpr std::__enable_if_t<std::chrono::__is_duration<_Tp>::value, std::chrono::time_point<_Clock, _ToDur> > std::chrono::time_point_cast(const time_point<_Clock, _Dur>&)’
 1030 |       time_point_cast(const time_point<_Clock, _Dur>& __t)
      |       ^~~~~~~~~~~~~~~
/usr/include/c++/13/bits/chrono.h:1030:7: note:   template argument deduction/substitution failed:
/usr/include/c++/13/mutex:192:62: note:   mismatched types ‘const std::chrono::time_point<_Clock, _Dur>’ and ‘const int’
  192 |           auto __s = chrono::time_point_cast<chrono::seconds>(__atime);
      |                      ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~^~~~~~~~~
[29/417] Compiling C++ object nntrainer/libn...ntrainer_nntrainer_graph_network_graph.cpp.o
*/
