#include <blas_neon_gemm_setting.h>

#define M_BLOCKING 192
// #define N_BLOCKING 2048
#define N_BLOCKING 9216
#define K_BLOCKING 384
void packing_a_v11(__fp16 *src, __fp16 *dst, unsigned int leading_dim,
                   unsigned int dim_first, unsigned int dim_second) {
  __fp16 *tosrc, *todst;
  todst = dst;
  unsigned int count_first, count_second, count_sub = dim_first;
  for (count_first = 0; count_sub > 23; count_first += 24, count_sub -= 24) {
    tosrc = src + count_first;
    for (count_second = 0; count_second < dim_second; count_second++) {
      vst1q_f16(todst, vld1q_f16(tosrc));
      vst1q_f16(todst + 8, vld1q_f16(tosrc + 8));
      vst1q_f16(todst + 16, vld1q_f16(tosrc + 16));
      tosrc += leading_dim;
      todst += 24;
    }
  }
  // adaptive loops
  for (; count_sub > 7; count_first += 8, count_sub -= 8) {
    tosrc = src + count_first;
    for (count_second = 0; count_second < dim_second; count_second++) {
      vst1q_f16(todst, vld1q_f16(tosrc));
      tosrc += leading_dim;
      todst += 8;
    }
  }
  for (; count_sub > 3; count_first += 4, count_sub -= 4) {
    tosrc = src + count_first;
    for (count_second = 0; count_second < dim_second; count_second++) {
      vst1_f16(todst, vld1_f16(tosrc));
      tosrc += leading_dim;
      todst += 4;
    }
  }
  for (; count_sub > 0; count_first += 1, count_sub -= 1) {
    tosrc = src + count_first;
    for (count_second = 0; count_second < dim_second; count_second++) {
      *todst = *tosrc;
      tosrc += leading_dim;
      todst++;
    }
  }
}

void packing_b_v11(__fp16 *src, __fp16 *dst, int leading_dim, int dim_first,
                   int dim_second) {
  // dim_first:K,dim_second:N
  __fp16 *tosrc1, *tosrc2, *todst;
  todst = dst;
  int count_first, count_second, count_sub = dim_second;
  for (count_second = 0; count_sub > 1; count_second += 2, count_sub -= 2) {
    tosrc1 = src + count_second * leading_dim;
    tosrc2 = tosrc1 + leading_dim;
    for (count_first = 0; count_first < dim_first; count_first++) {
      *todst = *tosrc1;
      tosrc1++;
      todst++;
      *todst = *tosrc2;
      tosrc2++;
      todst++;
    }
  }
  for (; count_sub > 0; count_second++, count_sub -= 1) {
    tosrc1 = src + count_second * leading_dim;
    for (count_first = 0; count_first < dim_first; count_first++) {
      *todst = *tosrc1;
      tosrc1++;
      todst++;
    }
  }
}

void hgemmv11_kernel_n_8(__fp16 *a_buffer, __fp16 *b_buffer, __fp16 *c_ptr,
                         unsigned int m, unsigned int K, unsigned int LDC,
                         float alpha) {
  unsigned int m_count, m_count_sub;
  unsigned int i, j, k;
  __fp16 *C = c_ptr;
  __fp16 sc0, sc1, sc2, sc3, sc4, sc5, sc6, sc7, sa, sb0, sb1, sb2, sb3, sb4,
    sb5, sb6, sb7;
  float16x4_t da, da0, da1, da2, db0, db1, db2, db3;
  float16x4_t dc00, dc10, dc20, dc30, dc40, dc50, dc60, dc70;
  float16x8_t valpha = vdupq_n_f16(alpha);
  float16x4_t dvalpha = vdup_n_f16(alpha);
  float16x8_t a, a0, a1, a2, b0, b1, b2, b3;
  float16x8_t c00, c01, c02, c10, c11, c12, c20, c21, c22, c30, c31, c32, c40,
    c41, c42, c50, c51, c52, c60, c61, c62, c70, c71, c72;
  float16x8_t c0, c1, c2, c3;
  __fp16 *ptr_packing_a, *ptr_packing_b0, *ptr_packing_b1, *ptr_packing_b2,
    *ptr_packing_b3;
  unsigned int k_start, k_end, K4;
  K4 = K & -4;
  k_end = K;
  k_start = 0;
  for (m_count_sub = m, m_count = 0; m_count_sub > (VL_FP16_TRIPLE - 1);
       m_count_sub -= VL_FP16_TRIPLE, m_count += VL_FP16_TRIPLE) {
    // call the micro kernel: m24n8;
    i = m_count;
    j = 0;
    ptr_packing_a = a_buffer + m_count * K;
    ptr_packing_b0 = b_buffer;
    ptr_packing_b1 = ptr_packing_b0 + 2 * K;
    ptr_packing_b2 = ptr_packing_b1 + 2 * K;
    ptr_packing_b3 = ptr_packing_b2 + 2 * K;
    macro_KERNEL_24xkx8_packing_v2();
  }
  for (; m_count_sub > (VL_FP16 - 1);
       m_count_sub -= VL_FP16, m_count += VL_FP16) {
    // call the micro kernel: m8n8;
    i = m_count;
    j = 0;
    ptr_packing_a = a_buffer + m_count * K;
    ptr_packing_b0 = b_buffer;
    ptr_packing_b1 = ptr_packing_b0 + 2 * K;
    ptr_packing_b2 = ptr_packing_b1 + 2 * K;
    ptr_packing_b3 = ptr_packing_b2 + 2 * K;
    macro_KERNEL_8xkx8_packing_v2();
  }
  for (; m_count_sub > (VL_FP16_HALF - 1);
       m_count_sub -= VL_FP16_HALF, m_count += VL_FP16_HALF) {
    // call the micro kernel: m2n8;
    i = m_count;
    j = 0;
    ptr_packing_a = a_buffer + m_count * K;
    ptr_packing_b0 = b_buffer;
    ptr_packing_b1 = ptr_packing_b0 + 2 * K;
    ptr_packing_b2 = ptr_packing_b1 + 2 * K;
    ptr_packing_b3 = ptr_packing_b2 + 2 * K;
    macro_KERNEL_4xkx8_packing_v2();
  }
  for (; m_count_sub > 0; m_count_sub -= 1, m_count += 1) {
    // call the micro kernel: m1n8;
    i = m_count;
    j = 0;
    ptr_packing_a = a_buffer + m_count * K;
    ptr_packing_b0 = b_buffer;
    ptr_packing_b1 = ptr_packing_b0 + 2 * K;
    ptr_packing_b2 = ptr_packing_b1 + 2 * K;
    ptr_packing_b3 = ptr_packing_b2 + 2 * K;
    macro_KERNEL_1xkx8_packing_v2();
  }
}

void hgemmv11_kernel_n_4(__fp16 *a_buffer, __fp16 *b_buffer, __fp16 *c_ptr,
                         unsigned int m, unsigned int K, unsigned int LDC,
                         float alpha) {
  unsigned int m_count, m_count_sub;
  unsigned int i, j, k;
  __fp16 *C = c_ptr;
  __fp16 sc0, sc1, sc2, sc3, sc4, sc5, sc6, sc7, sa, sb0, sb1, sb2, sb3, sb4,
    sb5, sb6, sb7;
  float16x4_t da, da0, da1, da2, db0, db1, db2, db3;
  float16x4_t dc00, dc10, dc20, dc30, dc40, dc50, dc60, dc70;
  float16x8_t valpha = vdupq_n_f16(alpha); // broadcast alpha to a 512-bit
                                           // vector
  float16x4_t dvalpha = vdup_n_f16(alpha); // broadcast alpha to a 128-bit
                                           // vector
  float16x8_t a, a0, a1, a2, b0, b1, b2, b3;
  float16x8_t c00, c01, c02, c10, c11, c12, c20, c21, c22, c30, c31, c32, c40,
    c41, c42, c50, c51, c52, c60, c61, c62, c70, c71, c72;
  float16x8_t c0, c1, c2, c3;
  __fp16 *ptr_packing_a, *ptr_packing_b;
  unsigned int k_start, k_end, K_;
  K_ = K & -4;
  k_end = K;
  k_start = 0;
  for (m_count_sub = m, m_count = 0; m_count_sub > (VL_FP16_TRIPLE - 1);
       m_count_sub -= VL_FP16_TRIPLE, m_count += VL_FP16_TRIPLE) {
    // call the micro kernel: m24n4;
    i = m_count;
    j = 0;
    ptr_packing_a = a_buffer + m_count * K;
    ptr_packing_b = b_buffer;
    macro_KERNEL_24xkx4_packing();
  }
  for (; m_count_sub > (VL_FP16 - 1);
       m_count_sub -= VL_FP16, m_count += VL_FP16) {
    // call the micro kernel: m8n4;
    i = m_count;
    j = 0;
    ptr_packing_a = a_buffer + m_count * K;
    ptr_packing_b = b_buffer;
    macro_KERNEL_8xkx4_HGEMM_packing();
  }
  for (; m_count_sub > 0; m_count_sub -= 1, m_count += 1) {
    i = m_count;
    j = 0;
    ptr_packing_a = a_buffer + m_count * K;
    ptr_packing_b = b_buffer;
    macro_KERNEL_1xkx4_packing()
  }
}

void hgemmv11_macro_kernel(__fp16 *a_buffer, __fp16 *b_buffer, unsigned int m,
                           unsigned int n, unsigned int k, __fp16 *C,
                           unsigned int LDC, float alpha) {
  unsigned int m_count, n_count, m_count_sub, n_count_sub;
  for (n_count_sub = n, n_count = 0; n_count_sub > (VL_FP16 - 1);
       n_count_sub -= VL_FP16, n_count += VL_FP16) {
    hgemmv11_kernel_n_8(a_buffer, b_buffer + n_count * k, C + n_count * LDC, m,
                        k, LDC, alpha);
  }
  for (; n_count_sub > (VL_FP16_HALF - 1);
       n_count_sub -= VL_FP16_HALF, n_count += VL_FP16_HALF) {
    hgemmv11_kernel_n_4(a_buffer, b_buffer + n_count * k, C + n_count * LDC, m,
                        k, LDC, alpha);
  }
  for (; n_count_sub > 1; n_count_sub -= 2, n_count += 2) {
    std::cout << "[ NYI ] Call the m layer with n=2\n";
  }
  for (; n_count_sub > 0; n_count_sub -= 1, n_count += 1) {
    std::cout << "[ NYI ] Call the m layer with n=1\n";
  }
}

void hgemmv11(unsigned int M, unsigned int N, unsigned int K, float alpha,
              __fp16 *A, unsigned int LDA, __fp16 *B, unsigned int LDB,
              float beta, __fp16 *C, unsigned int LDC) {
  ///@todo do for beta

  __fp16 *b_buffer =
    (__fp16 *)aligned_alloc(4096, K_BLOCKING * N_BLOCKING * sizeof(__fp16));
  __fp16 *a_buffer =
    (__fp16 *)aligned_alloc(4096, K_BLOCKING * M_BLOCKING * sizeof(__fp16));

  unsigned int m_count, n_count, k_count;
  unsigned int m_inc, n_inc, k_inc;
  for (n_count = 0; n_count < N; n_count += n_inc) {
    n_inc = (N - n_count > N_BLOCKING) ? N_BLOCKING : N - n_count;
    for (k_count = 0; k_count < K; k_count += k_inc) {
      k_inc = (K - k_count > K_BLOCKING) ? K_BLOCKING : K - k_count;
      packing_b_v11(B + k_count + n_count * LDB, b_buffer, LDB, k_inc, n_inc);
      for (m_count = 0; m_count < M; m_count += m_inc) {
        m_inc = (M - m_count > M_BLOCKING) ? M_BLOCKING : N - m_count;
        packing_a_v11(A + m_count + k_count * LDA, a_buffer, LDA, m_inc, k_inc);
        // macro kernel for packed A, B
        hgemmv11_macro_kernel(a_buffer, b_buffer, m_inc, n_inc, k_inc,
                              &C(m_count, n_count), LDC, alpha);
      }
    }
  }
  free(a_buffer);
  free(b_buffer);
}