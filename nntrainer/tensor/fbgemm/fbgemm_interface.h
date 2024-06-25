#include "./Fbgemm.h"
#include "./FbgemmPackMatrixB.h"
#include "./Utils.h"

template <typename T>
void Fbgemm_GEMM_interface(unsigned int btrans,const unsigned int k, const unsigned int n,
                           const float alpha,const T *B, unsigned int atrans,
                           const unsigned int m, float *A,const float beta, float *C,
                           unsigned int tid, unsigned int num_threads) ;

// template <typename T>
// void Fbgemm_GEMM_interface(unsigned int btrans,const unsigned int k, const unsigned int n,
//                            const float alpha,const T *B, unsigned int atrans,
//                            const unsigned int m, float *A,const float beta, float *C,
//                            unsigned int tid, unsigned int num_threads) {
// //   matrix_op_t _btrans = (matrix_op_t)btrans;
// //   const int _k = k;
// //   const int _n = n;
// //   const float _alpha = alpha;
// //   const T *_B = B;
// //   PackedGemmMatrixB<T> Bp(_btrans, _k, _n, _alpha, _B);

//   PackedGemmMatrixB<T> Bp(btrans, k, n, alpha, B);
//   cblas_gemm_compute((matrix_op_t)atrans, m, A, Bp, beta, C, tid, num_threads);
// }