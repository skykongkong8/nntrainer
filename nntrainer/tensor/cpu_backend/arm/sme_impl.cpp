#include <sme_impl.h>
// #include <sme_hwcap.h>
#include <arm_sme.h>
#include <arm_sve.h>

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <pthread.h>
#include <random>
#include <stddef.h>
#include <stdint.h>
#include <sys/auxv.h>
#include <thread>
#include <vector>

#ifndef AT_HWCAP2
#define AT_HWCAP2 26
#endif
#ifndef HWCAP2_SME
#define HWCAP2_SME                                                             \
  (1UL << 23) // 플랫폼에 따라 값이 다를 수 있음(커널 헤더 우선)
#endif

static inline bool has_sme() {
  unsigned long hw2 = getauxval(AT_HWCAP2);
  return (hw2 & HWCAP2_SME) != 0;
}

// 스트리밍 전용 함수(여기서는 ZA는 호출자에서 여닫음)
static void sme_hello_body() __arm_streaming {
  // ZA가 켜져 있어야만 유효
  asm volatile("zero {za}");
}

int hello_sme_function() {
  if (!has_sme()) {
    std::puts("SME not available. Falling back.");
    return 0;
  }
  // streaming Mode 와 Tile array를 모두 사용확인

  // 1) 스트리밍 모드 시작
  asm volatile("smstart sm");
  // 2) ZA 시작 (★ 중요)
  asm volatile("smstart za");

  // ZA 사용하는 본문
  sme_hello_body();

  // 3) ZA 종료
  asm volatile("smstop za");
  // 4) 스트리밍 모드 종료
  asm volatile("smstop sm");

  std::puts("SME hello passed (SM + ZA).");
  return 0;
}

static void scopy_kernel_sve(const float *__restrict x, float *__restrict y,
                             size_t n) __arm_streaming {
  size_t i = 0;
  for (; i < n;) {
    svbool_t pg = svwhilelt_b32((uint32_t)i, (uint32_t)n);
    svfloat32_t vx = svld1_f32(pg, &x[i]);
    svst1_f32(pg, &y[i], vx);
    i += svcntw();
  }
}

void scopy_sve(float *y, const float *x, size_t n) {
  asm volatile("smstart sm");
  scopy_kernel_sve(x, y, n);
  asm volatile("smstop sm");
}

static void saxpy_sve_kernel(float a, const float *__restrict x,
                             float *__restrict y, size_t n) __arm_streaming {
  size_t i = 0;
  const svfloat32_t va = svdup_f32(a);
  while (i < n) {
    svbool_t pg = svwhilelt_b32((uint32_t)i, (uint32_t)n);
    svfloat32_t vx = svld1_f32(pg, &x[i]);
    svfloat32_t vy = svld1_f32(pg, &y[i]);
    svfloat32_t vout = svmla_f32_m(pg, vy, vx, va); // y = a*x + y
    svst1_f32(pg, &y[i], vout);
    i += svcntw();
  }
}

void saxpy_sve(float a, const float *__restrict x, float *__restrict y,
               size_t n) {
  asm volatile("smstart sm");
  saxpy_sve_kernel(a, x, y, n);
  asm volatile("smstop sm");
}

static float sdot_sve_kernel(const float *__restrict x,
                             const float *__restrict y,
                             size_t n) __arm_streaming {
  size_t i = 0;
  svfloat32_t vacc = svdup_f32(0.0F);
  while (i < n) {
    svbool_t pg = svwhilelt_b32((uint32_t)i, (uint32_t)n);
    svfloat32_t vx = svld1_f32(pg, &x[i]);
    svfloat32_t vy = svld1_f32(pg, &y[i]);
    vacc = svmla_f32_m(pg, vacc, vx, vy); // acc += x * y
    i += svcntw();
  }
  return svaddv_f32(svptrue_b32(), vacc);
}

float sdot_sve(const float *__restrict x, const float *__restrict y, size_t n) {
  asm volatile("smstart sm");
  float ret = sdot_sve_kernel(x, y, n);
  asm volatile("smstop sm");
  return ret;
}

__attribute__((target("+sme2,+sme,+sve2"))) static float
sdot_sme2_kernel(const float *__restrict x, const float *__restrict y,
                 size_t n) __arm_streaming __arm_inout("za") {
  const uint32_t step = svcntw();
  svbool_t pg = svptrue_b32();

  // SVE 누산(정답)
  svfloat32_t vacc = svdup_f32(0.0f);

  // ZA는 성능 가속용으로 사용(필수 아님). sdot 결과는 vacc에서 반환.
  asm volatile("zero {za}");

  size_t i = 0;
  // 메인 루프: step 배수 구간은 ZA 경로 + SVE 누산 병행
  for (; i + step <= n; i += step) {
    svfloat32_t vx = svld1_f32(pg, &x[i]);
    svfloat32_t vy = svld1_f32(pg, &y[i]);

    // SVE 정답 누산
    vacc = svmla_f32_m(pg, vacc, vx, vy);

    // SME 사용: bx4 = {vy, 0, 0, 0}, ax1 = vx
    svfloat32_t z0 = svdup_f32(0.0f);
    svfloat32x4_t bx4 = svcreate4_f32(vy, z0, z0, z0);
    svmla_za32_vg1x4(/*row=*/0u, /*B(4)=*/bx4, /*A(1)=*/vx); // ZA not enabled?
  }

  // 테일(남은 < step)은 SVE로 마저 누산
  if (i < n) {
    svbool_t m = svwhilelt_b32((uint32_t)i, (uint32_t)n);
    svfloat32_t vx = svld1_f32(m, &x[i]);
    svfloat32_t vy = svld1_f32(m, &y[i]);
    vacc = svmla_f32_m(m, vacc, vx, vy);
  }

  // 최종 스칼라 반환: SVE 누산에서 읽음
  return svaddv_f32(svptrue_b32(), vacc);
}

__arm_new("za") float sdot_sme(const float *__restrict x,
                               const float *__restrict y, size_t n) {
  if (0) {
    // if (!has_sme()){
    return sdot_sve_kernel(x, y, n);
  } else {
    // asm volatile("smstart sm");
    asm volatile("smstart za");
    asm volatile("zero {za}");

    float ret = sdot_sme2_kernel(x, y, n);

    // asm volatile("smstop sm");
    asm volatile("smstop za");

    return ret;
  }
}

static void pack_sgemv_panel_f32(
  const float *__restrict A, int lda, const float *__restrict x, int incx,
  int M, int N,     // 전체 크기
  int m0, int n0,   // 블록 시작
  int &MB, int &NB, // [out] 실제 블록 크기(테일 대응)
  std::vector<float>
    &Ap, // [out] 크기 MB*NB, 레이아웃: 열-주도(column-major-by-block)
  std::vector<float> &Xp // [out] 크기 NB
) {
  const int VL = (int)svcntw();
  MB = (M - m0) < VL ? (M - m0) : VL;
  NB = (N - n0) < VL ? (N - n0) : VL;

  Ap.assign((size_t)MB * NB, 0.0f);
  Xp.assign(NB, 0.0f);

  // x 블록
  for (int jb = 0; jb < NB; ++jb) {
    Xp[jb] = x[(ptrdiff_t)(n0 + jb) * incx];
  }

  // A 블록을 열 기준으로 패킹: Ap[jb*MB + ib] = A[m0+ib, n0+jb]
  for (int jb = 0; jb < NB; ++jb) {
    const float *colp = A + (ptrdiff_t)(n0 + jb); // 행 m에 대해 오프셋 m*lda
    float *dst = Ap.data() + (size_t)jb * MB;
    for (int ib = 0; ib < MB; ++ib) {
      dst[ib] = A[(ptrdiff_t)(m0 + ib) * lda + (n0 + jb)];
    }
  }
}

__arm_new("za") static void sgemv_kernel_sme1_panel_f32(
  const float *__restrict Ap, // 크기 MB*NB, column-major-by-block
  const float *__restrict Xp, // 크기 NB
  int MB, int NB,
  float *__restrict y_partial // 크기 MB, [out]
  ) __arm_streaming {
  const int VL = (int)svcntw();
  // 유효 행/열 프레디케이트
  svbool_t pm = svwhilelt_b32((uint32_t)0, (uint32_t)MB);
  svbool_t pn = svwhilelt_b32((uint32_t)0, (uint32_t)NB);

  // ZA 초기화
  svzero_za();

  // 열 lane마다 외적 누적: ZA += (col_j) ⊗ (onehot_j * x[j])
  for (int jb = 0; jb < NB; ++jb) {
    // 행 MB개의 열 벡터 로드 (Ap는 column-major-by-block: jb*MB ..)
    const float *colj = Ap + (size_t)jb * MB;
    // col_j를 벡터화(행 방향)
    // 주: MB < VL일 수 있으므로 pm 마스크로 로드
    svfloat32_t vcol = svld1_f32(pm, colj);

    // x[jb]를 열 lane jb에만 배치하기 위한 컬럼 프레디케이트 생성
    // 방법: j-index 벡터와 상수 jb 비교
    svuint32_t jidx = svindex_u32(0, 1);
    svbool_t pcol = svcmpeq_u32(pn, jidx, svdup_u32((uint32_t)jb));

    // y-벡터는 상수 x[jb]를 전 레인에 복제해도, pcol로 컬럼을 단일 lane로 제한
    svfloat32_t vy = svdup_f32(Xp[jb]);

    // ZA 누적: 타일 0 사용
    svmopa_za32_m(0 /*tile*/, pm /*rows mask*/, pcol /*cols mask*/, vcol, vy);
  }

  // ZA의 각 행 슬라이스를 배열로 꺼내서, 유효 NB 부분만 합해 y_partial에 기록
  // svst1_hor_za32(tile=0, pn, buf, trow) 가 열 방향으로 VL개를 저장
  for (int trow = 0; trow < MB; ++trow) {
    alignas(64) float rowbuf_max[256]; // VL 최대치(실기기 VL<=256 bytes/32
                                       // lanes f32 등). 보수적으로 충분히 크게.
    // 안전하게 동적 크기 처리: 우선 VL 길이만큼 저장 후 NB만 합산
    // 실제 저장은 pn 프레디케이트(0..NB-1)만 유효 데이터
    svst1_hor_za32(0, trow, pn, rowbuf_max);
    // svst1_hor_za32(0, pn, rowbuf_max, trow);

    // NB 유효 구간만 SVE로 합산
    // 로드 시 pn 사용
    svfloat32_t vrow = svld1_f32(pn, rowbuf_max);
    float sum = svaddv_f32(pn, vrow); // 마스크 기반 수평합
    y_partial[trow] += sum;
  }
}

// 3) 상위 진입점: y := alpha*A*x + beta*y
// 전체 N을 VL 컬럼 블록으로 순회하며, 각 블록을 pack → 커널 호출 → y 누적
// extern "C" void sgemv_sme(
void sgemv_sme(int M, int N, float alpha, const float *__restrict A, int lda,
               const float *__restrict x, int incx, float beta,
               float *__restrict y, int incy) {
  if (M <= 0 || N <= 0)
    return;

  // y 스케일(beta) 선반영
  if (beta != 0.0f) {
    for (int i = 0; i < M; ++i) {
      y[(ptrdiff_t)i * incy] *= beta;
    }
  }

  const int VL = (int)svcntw();
  std::vector<float> Ap; // 패널 버퍼
  std::vector<float> Xp; // x 블록 버퍼
  std::vector<float> ypart;
  ypart.resize((size_t)((M < VL) ? M : VL));

  for (int m0 = 0; m0 < M; m0 += VL) {
    const int MB = ((M - m0) < VL) ? (M - m0) : VL;

    // y 부분합 초기화
    std::fill(ypart.begin(), ypart.begin() + MB, 0.0f);

    for (int n0 = 0; n0 < N; n0 += VL) {
      int MB_blk = 0, NB_blk = 0;
      // 현재 (m0..m0+MB-1) x (n0..n0+NB-1) 패널 패킹
      pack_sgemv_panel_f32(A, lda, x, incx, M, N, m0, n0, MB_blk, NB_blk, Ap,
                           Xp);

      // 커널 호출: ypart += A_panel * x_block (ZA 활용)
      sgemv_kernel_sme1_panel_f32(Ap.data(), Xp.data(), MB_blk, NB_blk,
                                  ypart.data());
    }

    // alpha 스케일 적용 후 y에 누적
    if (alpha != 1.0f) {
      for (int ib = 0; ib < MB; ++ib)
        ypart[ib] *= alpha;
    }
    for (int ib = 0; ib < MB; ++ib) {
      y[(ptrdiff_t)(m0 + ib) * incy] += ypart[ib];
    }
  }
}
