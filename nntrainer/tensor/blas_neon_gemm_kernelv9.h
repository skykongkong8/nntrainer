// #include <blas_neon_gemm_setting.h>

// #define M_BLOCKING 192
// #define N_BLOCKING 2048
// #define K_BLOCKING 384

// void packing_a_k9(__fp16 *src, __fp16 *dst, int leading_dim, int dim_first,
//                   int dim_second) {
//   // dim_first: M, dim_second: K
//   __fp16 *tosrc, *todst; // cache-using
//   todst = dst;
//   int count_first, count_second, count_sub = dim_first;
//   for (count_first = 0; count_sub > 7; count_first += 8, count_sub -= 8) {
//     tosrc = src + count_first;
//     for (count_second = 0; count_second < dim_second; count_second++) {
//       vst1q_f16(todst, vld1q_f16(tosrc));
//       tosrc += leading_dim;
//       todst += 8;
//     }
//   }
//   for (; count_sub > 3; count_first += 4, count_sub -= 4) {
//     tosrc = src + count_first;
//     for (count_second = 0; count_second < dim_second; count_second++) {
//       vst1_f16(todst, vld1_f16(tosrc));
//       tosrc += leading_dim;
//       todst += 4;
//     }
//   }
//   for (; count_sub > 0; count_first += 1, count_sub -= 1) {
//     tosrc = src + count_first;
//     for (count_second = 0; count_second < dim_second; count_second++) {
//       *todst = *tosrc;
//       tosrc += leading_dim;
//       todst++;
//     }
//   }
// }

// void packing_b_k9(__fp16 *src, __fp16 *dst, int leading_dim, int dim_first,
//                   int dim_second) {
//   // dim_first:K,dim_second:N
//   __fp16 *tosrc1, *tosrc2, *tosrc3, *tosrc4, *todst;
//   todst = dst;
//   int count_first, count_second;
//   for (count_second = 0; count_second < dim_second; count_second += 4) {
//     tosrc1 = src + count_second * leading_dim;
//     tosrc2 = tosrc1 + leading_dim;
//     tosrc3 = tosrc2 + leading_dim;
//     tosrc4 = tosrc3 + leading_dim;
//     for (count_first = 0; count_first < dim_first; count_first++) {
//       *todst = *tosrc1;
//       tosrc1++;
//       todst++;
//       *todst = *tosrc2;
//       tosrc2++;
//       todst++;
//       *todst = *tosrc3;
//       tosrc3++;
//       todst++;
//       *todst = *tosrc4;
//       tosrc4++;
//       todst++;
//     }
//   }
// }

// static inline void hgemmv9_opt(unsigned int M, unsigned int N, unsigned int K,
//                                float alpha, const __fp16 *A, unsigned int LDA,
//                                const __fp16 *B, unsigned int LDB, float beta,
//                                __fp16 *C, unsigned int LDC) {
//   unsigned int i, j, k;
//   /// @todo do for beta
//   // if (beta != 1.0) return;
//   for (i = 0; i < M; i++) {
//     for (j = 0; j < N; j++) {
//       __fp16 tmp = C(i, j);
//       // float tmp=C(i,j);
//       for (k = 0; k < K; k++) {
//         tmp += alpha * A(i, k) * B(k, j);
//       }
//       C(i, j) = tmp;
//     }
//   }
// }

// void hgemmv9_macro_kernel(unsigned int M, unsigned int N, unsigned int K,
//                           float alpha, __fp16 *A, unsigned int LDA, __fp16 *B,
//                           unsigned int LDB, __fp16 *C, unsigned int LDC) {
//   unsigned int i, j, k;
//   /// @note beta is already processed before
//   unsigned int M_ = M & -VL_FP16, N_ = N & -VL_FP16_HALF,
//                K_ = K & -VL_FP16_HALF;
//   unsigned int k_end = K;
//   auto ptr_packing_a = A;
//   auto ptr_packing_b = B;
//   /// @todo do for alpha
//   float16x8_t valpha = vdupq_n_f16(1.f); // broadcast alpha to a 256-bit vector
//   float16x8_t a, b0, b1, b2, b3;
//   float16x8_t c0, c1, c2, c3;

//   //   float16x4_t a_, b0_, b1_, b2_, b3_;
//   //   float16x4_t c0_, c1_, c2_, c3_;

//   for (i = 0; i < M_; i += VL_FP16) {
//     for (j = 0; j < N_; j += VL_FP16_HALF) {
//       ptr_packing_a = A + i * K;
//       ptr_packing_b = B + j * K;
//       macro_KERNEL_8xkx4_HGEMM_packing();
//     }
//   }
//   for (i = 0; i < M_; i += VL_FP16_HALF) {
//     for (j = 0; j < N_; j += VL_FP16_HALF) {
//       //   ptr_packing_a = A + i * K;
//       //   ptr_packing_b = B + j * K;
//       //   macro_KERNEL_4xkx4_HGEMM_packing();
//       // std::cout << "RESIDUAL BLOCKS!\n";
//     }
//   }
//   if (M_ == M && N_ == N)
//     return;
//   // boundary conditions
//   if (M_ != M)
//     hgemmv9_opt(M - M_, N, K, alpha, A + M_, LDA, B, LDB, 1.0, &C(M_, 0), LDC);
//   if (N_ != N)
//     hgemmv9_opt(M_, N - N_, K, alpha, A, LDA, &B(0, N_), LDB, 1.0, &C(0, N_),
//                 LDC);
// }

// void hgemmv9(unsigned int M, unsigned int N, unsigned int K, float alpha,
//              __fp16 *A, unsigned int LDA, __fp16 *B, unsigned int LDB,
//              float beta, __fp16 *C, unsigned int LDC) {
//   ///@todo do for beta

//   __fp16 *b_buffer =
//     (__fp16 *)aligned_alloc(4096, K_BLOCKING * N_BLOCKING * sizeof(__fp16));
//   __fp16 *a_buffer =
//     (__fp16 *)aligned_alloc(4096, K_BLOCKING * M_BLOCKING * sizeof(__fp16));

//   unsigned int m_count, n_count, k_count;
//   unsigned int m_inc, n_inc, k_inc;
//   for (n_count = 0; n_count < N; n_count += n_inc) {
//     n_inc = (N - n_count > N_BLOCKING) ? N_BLOCKING : N - n_count;
//     for (k_count = 0; k_count < K; k_count += k_inc) {
//       k_inc = (K - k_count > K_BLOCKING) ? K_BLOCKING : K - k_count;
//       packing_b_k9(B + k_count + n_count * LDB, b_buffer, LDB, k_inc, n_inc);
//       for (m_count = 0; m_count < M; m_count += m_inc) {
//         m_inc = (M - m_count > M_BLOCKING) ? M_BLOCKING : N - m_count;
//         packing_a_k9(A + m_count + k_count * LDA, a_buffer, LDA, m_inc, k_inc);
//         // macro kernel: to compute C += A_tilt * B_tilt
//         hgemmv9_macro_kernel(m_inc, n_inc, k_inc, alpha, &A(m_count, k_count),
//                              LDA, &B(k_count, n_count), LDB,
//                              &C(m_count, n_count), LDC);
//       }
//     }
//   }
//   free(a_buffer);
//   free(b_buffer);
// }