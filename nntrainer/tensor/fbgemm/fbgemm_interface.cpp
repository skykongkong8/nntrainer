#include "./fbgemm_interface.h"

template <>
void Fbgemm_GEMM_interface(unsigned int btrans,const unsigned int k, const unsigned int n,
                           const float alpha, const uint16_t *B, unsigned int atrans,
                           const unsigned int m, float *A,const float beta, float *C,
                           unsigned int tid, unsigned int num_threads) {
  matrix_op_t _btrans = (matrix_op_t)btrans;
  matrix_op_t _atrans = (matrix_op_t)atrans;
//   const int _k = k;
//   const int _n = n;
//   const float _alpha = alpha;
//   const T *_B = B;
//   PackedGemmMatrixB<T> Bp(_btrans, _k, _n, _alpha, _B);

  PackedGemmMatrixB<uint16_t> Bp(_btrans, k, n, alpha, B);
  cblas_gemm_compute<uint16_t>(_atrans, m, A, Bp, beta, C, tid, num_threads);
}