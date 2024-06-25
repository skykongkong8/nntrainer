/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

/**
 * Top level include file for FBGEMM.
 */
#include <cassert>
#include <cmath>
#include <limits>
#include <memory>
#include <type_traits>
#include "./FbgemmBuild.h"
#include "./Types.h"
#include "./Utils.h"
#include "./UtilsAvx2.h"

#ifdef __clang__
// clang-format off
#define FBGEMM_PUSH_WARNING _Pragma("GCC diagnostic push")
#define FBGEMM_DISABLE_WARNING_INTERNAL2(warningName) #warningName
#define FBGEMM_DISABLE_WARNING(warningName) \
  _Pragma(                                     \
      FBGEMM_DISABLE_WARNING_INTERNAL2(GCC diagnostic ignored warningName))
#define FBGEMM_PUSH_WARNING_AND_DISABLE(warningName) \
  _Pragma("GCC diagnostic push") \
  _Pragma(                                     \
      FBGEMM_DISABLE_WARNING_INTERNAL2(GCC diagnostic ignored warningName))
#define FBGEMM_POP_WARNING _Pragma("GCC diagnostic pop")
// clang-format on
#else
#define FBGEMM_PUSH_WARNING
#define FBGEMM_DISABLE_WARNING(NAME)
#define FBGEMM_PUSH_WARNING_AND_DISABLE(NAME)
#define FBGEMM_POP_WARNING
#endif

// Turning on this option will print out time breakdown of each stage (e.g.,
// input packing, the main GEMM kernel, each output processing pipeline).
// Please note that currently this option won't report accurate timing if
// multiple threads are used.
// #define FBGEMM_MEASURE_TIME_BREAKDOWN

#ifdef FBGEMM_MEASURE_TIME_BREAKDOWN
#include <chrono>
#include <iostream>
extern double packing_time;
extern double computing_time;
extern double kernel_time;
extern double postprocessing_time;
extern double run_time;
#endif


/**
 * @brief Templatized struct for packing parameters for A and B matrices.
 *
 * @tparam T input type
 * @tparam accT the type used for accumulation
 * @tparam instSet anyarch/avx2/avx512
 * @tparam int8Type an auxiliary template parameter to specialize for 8-bit
 *                  input types.
 */
template <
    typename T,
    typename accT,
    inst_set_t instSet,
    typename int8Type = void>
struct PackingTraits;

// type specialized implementation in an include file
#include "./PackingTraits-inl.h"

/**
 * @brief Base class for packing matrices for higher GEMM performance.
 *
 * Matrix is tiled into blockRows() * blockCols() blocks.
 * Each block is with size blockRowSize() * blockColSize().
 * This class is designed using CRTP
 * (https://en.wikipedia.org/wiki/Curiously_recurring_template_pattern)
 *
 * @tparam PT actual packing type, e.g., PackAWithRowOffset
 */
template <typename PT, typename inpType, typename accType = std::int32_t>
class PackMatrix {
 public:
  PackMatrix() = delete; // no default constructor
  PackMatrix(const PackMatrix&) = delete; // no copy
  PackMatrix& operator==(const PackMatrix&) = delete; // no copy
  PackMatrix(PackMatrix&&) = delete; // no move
  PackMatrix& operator==(PackMatrix&& rhs) noexcept = delete; // no move

  /**
   * @param rows total number of rows in the matrix
   *             (packed rows can be less than rows).
   * @param cols total number of columns in the matrix
   * @param pmat A buffer to contain the packed matrix.
   *             If nullptr, a buffer owned by PackMatrix will be allocated
   *             internally to contain the packed matrix.
   *             For non-constant matrices like activation matrices, the client
   *             code may want to pass a pre-allocated pmat to avoid the
   *             overhead of internal memory allocation everytime a PackMatrix
   *             is constructed. The client code can query how big patm should
   *             be with packedBufferSize function.
   * @param groups when groups > 1, we compute groups number of GEMMs each
   *               multiplies A.rows by A.cols/A.groups matrix with
   *               B.rows/B.groups by B.cols matrix (in conventional BLAS
   *               terminology, this is a batched GEMM but we use the name group
   *               to follow deep learning terminology). The result matrix has
   *               dimension A.rows by B.cols*B.groups .
   *               A.groups must be same as B.groups, A.groups must divide
   *               A.cols, and B.groups must divide B.rows and C.cols.
   */
  PackMatrix(
      std::int32_t rows,
      std::int32_t cols,
      inpType* pmat,
      int groups = 1,
      const BlockingFactors* params = nullptr);

  /**
   * @return true usually when the matrix is constant matrix (e.g., weight
   *         matrices) that can be prepacked
   */
  bool isPrePacked() const {
    return static_cast<const PT*>(this)->isPrePacked();
  }

  /**
   * @return true if this is the first input matrix in GEMM (i.e., A in C = A *
   *         B)
   */
  static constexpr bool isA() {
    return PT::isA();
  }

  /**
   * @brief The size of the buffer used for packing (The size is in number of
   *        elements).
   *
   * rows and cols are only used for fully packing, i.e., for B matrix.  The
   * client code can use this function to query how big the buffer used for
   * packing should be.
   */
  static int packedBufferSize(
      int rows = 0,
      int cols = 0,
      const BlockingFactors* params = nullptr);

  FBGEMM_PUSH_WARNING_AND_DISABLE("-Winfinite-recursion")
  /**
   * @return Pointer to a buffer containing row offset results. Some packing
   *         objects fuse row offset computation for later requantization step.
   */
  std::int32_t* getRowOffsetBuffer() const {
    return static_cast<const PT*>(this)->getRowOffsetBuffer();
  }
  FBGEMM_POP_WARNING

  FBGEMM_PUSH_WARNING_AND_DISABLE("-Winfinite-recursion")
  /**
   * @brief When k loop is also tiled/blocked, this function is used to check if
   * have executed computations for the last k block so that we can perform
   *        post-GEMM operations.
   */
  bool isThisLastKBlock(int block_id) const {
    return static_cast<const PT*>(this)->isThisLastKBlock(block_id);
  }
  FBGEMM_POP_WARNING

  /**
   * @brief Actual packing of a block of the source matrix in pmat buffer.
   */
  void pack(const block_type_t& block) {
    static_cast<PT*>(this)->pack(block);
  }

  std::int32_t numRows() const {
    return nrows_;
  }

  std::int32_t numCols() const {
    return ncols_;
  }

  /**
   * @return The number of rows in each block
   */
  std::int32_t blockRowSize() const {
    return brow_;
  }

  /**
   * @return The number of columns in each block
   */
  std::int32_t blockColSize() const {
    return bcol_;
  }

  /**
   * @return The number of blocks along rows
   */
  std::int32_t blockRows() const {
    return nbrow_;
  }

  /**
   * @return The number of blocks along columns
   */
  std::int32_t blockCols() const {
    return nbcol_;
  }

  /**
   * @return The number of the rows in the currently packed block of a matrix.
   *         For pre-packed (i.e., fully-packed), it's equal to the total number
   * of rows.
   */
  std::int32_t numPackedRows() const {
    return packedBlock_.row_size;
  }

  /**
   * @return The number of columns in the currently packed block of a matrix.
   *         For pre-packed (i.e., fully-packed), it's equal to the number of
   * columns.
   */
  std::int32_t numPackedCols() const {
    return packedBlock_.col_size;
  }

  /**
   * @return The first row of the block we're working on.
   */
  std::int32_t packedRowStart() const {
    return packedBlock_.row_start;
  }

  /**
   * @return The first column of the block we're working on.
   */
  std::int32_t packedColStart() const {
    return packedBlock_.col_start;
  }

  /**
   * @return The beginning of (rowBlockNum, colBlockNum)th block
   */
  inpType* getBuf(std::int32_t rowBlockNum = 0, std::int32_t colBlockNum = 0) {
    return buf_ + blockRowSize() * blockColSize() * rowBlockNum +
        blockRowSize() * blockColSize() * blockCols() * colBlockNum;
  }

  /**
   * @brief Print the packed block.
   */
  void printPackedMatrix(std::string name) {
    static_cast<PT*>(this)->printPackedMatrix(name);
  }

  /**
   * @return The number of rows in the last row block.
   */
  std::int32_t lastBrow() const {
    return last_brow_;
  }

  /**
   * @return The number of columns in the last column block.
   */
  std::int32_t lastBcol() const {
    return last_bcol_;
  }

  int numGroups() const {
    return G_;
  }

  /**
   * @return True if the last column block has fewer columns than the block
   *         size.
   */
  bool isThereColRemainder() const {
    return last_bcol_ != blockColSize();
  }

  virtual ~PackMatrix() {
    if (bufAllocatedHere_) {
      fbgemmAlignedFree(buf_);
    }
  }

 protected:
  /**
   * Set which block we're packing
   */
  void packedBlock(const block_type_t& block) {
    packedBlock_ = block;
    nbrow_ = (numPackedRows() + blockRowSize() - 1) / blockRowSize();
    nbcol_ = (numPackedCols() + blockColSize() - 1) / blockColSize();

    last_brow_ = ((numPackedRows() % blockRowSize()) == 0)
        ? blockRowSize()
        : (numPackedRows() % blockRowSize());
    last_bcol_ = ((numPackedCols() % blockColSize()) == 0)
        ? blockColSize()
        : (numPackedCols() % blockColSize());
  }

  inpType* buf_;
  std::int32_t brow_; ///< the number of rows in each block
  std::int32_t bcol_; ///< the number of columns in each block
  std::int32_t nbrow_; ///< the number of blocks along rows
  std::int32_t nbcol_; ///< the number of blocks along columns
  bool bufAllocatedHere_{false};
  const BlockingFactors*
      blocking_params; ///< MCB, KCB, NCB, MR, NR, NR_MIN, ROW_INTERLEAVE;

 private:
  std::int32_t nrows_, ncols_;
  int G_;
  block_type_t packedBlock_; ///< The block in the source matrix just packed
  std::int32_t last_brow_, last_bcol_;
};

/**
 * @brief Matrix packed for the first input matrix in GEMM (usually
 *        activation).  The source matrix is already quantized. Default
 * accumulation type is int32.
 */
template <typename T, typename accT = std::int32_t>
class  PackAMatrix final
    : public PackMatrix<PackAMatrix<T, accT>, T, accT> {
 public:
  using This = PackAMatrix<T, accT>;
  using BaseType = PackMatrix<This, T, accT>;
  using inpType = T;
  using accType = accT;

  PackAMatrix() = delete; // no default constructor

  PackAMatrix(
      matrix_op_t trans,
      std::int32_t nRow,
      std::int32_t nCol,
      const inpType* smat,
      std::int32_t ld,
      inpType* pmat = nullptr,
      int groups = 1,
      const BlockingFactors* params = nullptr);

  /**
   * Activation matrices are not constant so cannot amortize the cost of
   * pre-packing.
   */
  bool isPrePacked() const {
    return false;
  }

  /**
   * @return True if this is used as A matrix.
   */
  static constexpr bool isA() {
    return true;
  }

  /**
   * @return A pointer to the row offset buffer. There is no row offset buffer
   *         calculations with this packing class, hence, it returns nullptr.
   */
  std::int32_t* getRowOffsetBuffer() const {
    return nullptr;
  }

  /**
   * @return Offset of the element in the packed matrix that was at (i, j) in
   *         the source matrix.
   */
  std::int32_t addr(std::int32_t i, std::int32_t j) const;

  /**
   * @brief Packs a block of source matrix into pmat buffer.
   */
  void pack(const block_type_t& block);

  /**
   * @brief Print the packed block.
   */
  void printPackedMatrix(std::string name);

 private:
  matrix_op_t trans_;
  const T* smat_;
  std::int32_t ld_;
  std::int32_t row_interleave_B_;
};

/**
 * @brief Matrix packed for the second input matrix in GEMM (usually weight).
 *        The source matrix is already quantized. Default accumulation
 *        type is int32.
 */
template <typename T, typename accT = std::int32_t>
class  PackBMatrix final
    : public PackMatrix<PackBMatrix<T, accT>, T, accT> {
 public:
  using This = PackBMatrix<T, accT>;
  using BaseType = PackMatrix<This, T, accT>;
  using inpType = T;
  using accType = accT;

  PackBMatrix() = delete; // no default constructor

  /**
   * @param groups if > 1 and trans == NoTranspose, smat is nRow x nCol with
   *               groups are vertically concatenated: each group is
   *               (nRow / groups) x nCol .
   *               if > 1 and trans == Transpose, smat is (nCol * groups) x
   *               (nRow / groups) with groups are horizontally concatenated:
   *               each group is nCol x (nRow / groups) . Each group is
   *               transposed and vertically concatenated to match with the
   *               NoTranspose case.
   */
  PackBMatrix(
      matrix_op_t trans,
      std::int32_t nRow,
      std::int32_t nCol,
      const inpType* smat,
      std::int32_t ld,
      inpType* pmat = nullptr,
      int groups = 1,
      const BlockingFactors* params = nullptr);

  /**
   * Weight matrices are usually constant so worth pre-packing.
   */
  bool isPrePacked() const {
    return true;
  }

  /**
   * @return True if to be used as A matrix, False otherwise.
   */
  static constexpr bool isA() {
    return false;
  }

  /**
   * @brief When k loop is also tiled/blocked, this function is used to check if
   * have executed computations for the last k block so that we can perform
   *        post-GEMM operations.
   */
  bool isThisLastKBlock(int block_id) const {
    return (BaseType::blockRows() - 1) == block_id;
  }

  /**
   * @return Offset of the element in the packed matrix that was at (i, j) in
   *         the source matrix.
   */
  std::int32_t addr(std::int32_t i, std::int32_t j) const;

  /**
   * @brief Packs a block of source matrix into pmat buffer. The blocking
   *        parameters are needed to compute the buffer size of each group.
   *        It will use default blocking parameters if params is not provided.
   */
  void pack(const block_type_t& block, const BlockingFactors* params = nullptr);

  /**
   * @brief Print the packed block.
   */
  void printPackedMatrix(
      std::string name,
      const BlockingFactors* params = nullptr);

  /**
   * @return true if meta information like matrix shape is the same.
   */
  bool metaEquals(const PackBMatrix<T, accT>& that) const;
  /**
   * @return true if matrices are the same.
   */
  bool equals(const PackBMatrix<T, accT>& that) const;

  /**
   * @brief Unpack pmat buffer to the origin_buf (Used for the serialization to
   * recover weight matrix).
   */
  void unpack(T* origin_buf, const BlockingFactors* params = nullptr);

  ~PackBMatrix() {}

 private:
  matrix_op_t trans_;
  const T* smat_;
  std::int32_t ld_;
  std::int32_t row_interleave_;

  /**
   * @brief Internal function performing both pack & unpack
   */
  void pack_unpack_(
      const block_type_t& block,
      T* unpack_buf,
      T* pack_buf,
      bool ispack,
      const BlockingFactors* params = nullptr);
};


/*
 *
 * Post Processing of outputs
 *
 */

/**
 * @brief Does nothing. NoOp. Used as the last operation in the output
 *        processing pipeline.
 *
 */
template <typename outT = std::uint8_t, typename inT = std::uint8_t>
class  DoNothing {
 public:
  using outType = outT;
  using inpType = inT;
  DoNothing() {}
  template <inst_set_t instSet>
  int f(
      outType* /* unused */,
      inpType* /* unused */,
      const block_type_t& /* unused */,
      int /* unused */,
      int /* unused */) const {
    return 0;
  }
};

/**
 * @brief Copy data pointed by inp ptr to out ptr when
 *        inp ptr and out ptr are not the same.
 *        inp buffer: row and column start points: (0, 0)
 *        output buffer: row and column start points:
 *        (block.row_start, block.col_start)
 *
 * This is the output processing stage that should passed when there is no
 * requantization and output is required in the same format as internal buffer
 * used for accumulation.
 */
template <
    typename outT = std::int32_t,
    typename inT = std::int32_t,
    typename nextOPType = DoNothing<outT, outT>>
class memCopy {
 public:
  using outType = outT;
  using inpType = inT;
  explicit memCopy(nextOPType& nextop) : nextop_(nextop) {}
  template <inst_set_t instSet>
  inline int f(
      outType* out,
      inpType* inp,
      const block_type_t& block,
      int ld_out,
      int ld_in) const;

 private:
  nextOPType& nextop_;
};

/**
 * @brief Perform scaling on accumulated data.
 */
template <
    typename outT = std::int32_t,
    typename inT = std::int32_t,
    typename nextOPType = DoNothing<outT, outT>>
class ScaleOP {
 public:
  using outType = outT;
  using inpType = inT;
  explicit ScaleOP(inpType scalingFactor) : scalingFactor_(scalingFactor) {}

  template <inst_set_t instSet>
  inline int f(
      outType* out,
      inpType* inp,
      const block_type_t& block,
      int ld_out,
      int ld_in) const;

 private:
  inpType scalingFactor_;
};


// type specialized implementation in an include file
// #include "./OutputProcessing-inl.h"

/*
 *
 * ####### GEMM related functions #######
 *
 */

/**
 * Matrix B must be prepacked. For matrix A, packA.pack function is called to
 * pack it.
 *
 * @tparam packingAMatrix processing of A matrix while packing,
 *                        e.g., PackAWithQuantRowOffset
 *
 * @tparam packingBMatrix processing of B matrix while packing,
 *                        e.g.,  pre-multiply by alpha
 * @tparam cT data type of C matrix
 * @tparam processOutputType further processing of outputs, e.g., Relu
 */
template <
    typename packingAMatrix,
    typename packingBMatrix,
    typename cT,
    typename processOutputType>
 void fbgemmPacked(
    PackMatrix<
        packingAMatrix,
        typename packingAMatrix::inpType,
        typename packingAMatrix::accType>& packA,
    PackMatrix<
        packingBMatrix,
        typename packingBMatrix::inpType,
        typename packingBMatrix::accType>& packB,
    cT* C,
    std::int32_t* C_buffer,
    std::uint32_t ldc,
    const processOutputType& outProcess,
    int thread_id,
    int num_threads,
    const BlockingFactors* blocking_params = nullptr);

/**
 * @brief Are we running on a fbgemm supported cpu?
 */
bool fbgemmSupportedCPU();
