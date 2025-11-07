// sgemv_sme1_allinone.cpp
#include <cstddef>
#include <cstdint>
#include <vector>
#include <arm_sve.h>
#include <arm_sme.h>

// 제약(단순화):
// - 행렬 A: row-major, lda >= N
// - incx == 1, incy == 1 (필요 시 gather/compact 경로 확장)
// - SME2 불사용. SME1 전용 인스트린식(svzero_za, svmopa_za32_m, svst1_hor_za32).

// 1) 패킹: (m0..m0+MB-1) x (n0..n0+NB-1) 블록을
// A_rows_by_col 형태로 재배열: 각 열 j의 MB개 행이 연속 저장.
// x 블록도 연속 복사.
// 목적: 커널에서 '열 벡터(행 방향 연속)'를 한 번에 로드 가능하게.
static void pack_sgemv_panel_f32(
    const float* __restrict A, int lda,
    const float* __restrict x, int incx,
    int M, int N, // 전체 크기
    int m0, int n0, // 블록 시작
    int& MB, int& NB, // [out] 실제 블록 크기(테일 대응)
    std::vector<float>& Ap, // [out] 크기 MB*NB, 레이아웃: 열-주도(column-major-by-block)
    std::vector<float>& Xp // [out] 크기 NB
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
        const float* colp = A + (ptrdiff_t)(n0 + jb); // 행 m에 대해 오프셋 m*lda
        float* dst = Ap.data() + (size_t)jb * MB;
        for (int ib = 0; ib < MB; ++ib) {
            dst[ib] = A[(ptrdiff_t)(m0 + ib) * lda + (n0 + jb)];
        }
    }
}

// 2) SME1 커널: ZA를 이용해 패널(행 MB x 열 NB)과 x블록(NB)을 처리해
// 행 MB 개의 부분 누적값(partial sums)을 반환.
// 구현 전략:
// - ZA를 0으로 초기화
// - 블록의 각 열 lane jb에 대해, 열 벡터(col_j)와 스칼라 x[jb]만을
// ZA의 동일 열 lane jb에 누적하도록 컬럼 프레딕트 생성 → svmopa_za32_m 호출을 NB번 반복
// - 최종적으로 ZA의 각 행 슬라이스를 읽고(행 길이 = VL), 유효 NB 구간만 수평합해 y_partial(행 MB) 생성
static void sgemv_kernel_sme1_panel_f32(
    const float* __restrict Ap, // 크기 MB*NB, column-major-by-block
    const float* __restrict Xp, // 크기 NB
    int MB, int NB,
    float* __restrict y_partial // 크기 MB, [out]
)
__arm_streaming __arm_new("za")
{
    const int VL = (int)svcntw();
    // 유효 행/열 프레디케이트
    svbool_t pm = svwhilelt_b32((uint32_t)0, (uint32_t)MB);
    svbool_t pn = svwhilelt_b32((uint32_t)0, (uint32_t)NB);

    // ZA 초기화
    svzero_za();

    // 열 lane마다 외적 누적: ZA += (col_j) ⊗ (onehot_j * x[j])
    for (int jb = 0; jb < NB; ++jb) {
        // 행 MB개의 열 벡터 로드 (Ap는 column-major-by-block: jb*MB ..)
        const float* colj = Ap + (size_t)jb * MB;
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
        alignas(64) float rowbuf_max[256]; // VL 최대치(실기기 VL<=256 bytes/32 lanes f32 등). 보수적으로 충분히 크게.
        // 안전하게 동적 크기 처리: 우선 VL 길이만큼 저장 후 NB만 합산
        // 실제 저장은 pn 프레디케이트(0..NB-1)만 유효 데이터
        svst1_hor_za32(0, pn, rowbuf_max, trow);

        // NB 유효 구간만 SVE로 합산
        // 로드 시 pn 사용
        svfloat32_t vrow = svld1_f32(pn, rowbuf_max);
        float sum = svaddv_f32(pn, vrow); // 마스크 기반 수평합
        y_partial[trow] += sum;
    }
}

// 3) 상위 진입점: y := alpha*A*x + beta*y
// 전체 N을 VL 컬럼 블록으로 순회하며, 각 블록을 pack → 커널 호출 → y 누적
extern "C" void sgemv_sme(
    int M, int N,
    float alpha,
    const float* __restrict A, int lda,
    const float* __restrict x, int incx,
    float beta,
    float* __restrict y, int incy)
{
    if (M <= 0 || N <= 0) return;

    // y 스케일(beta) 선반영
    if (beta != 1.0f) {
        for (int i = 0; i < M; ++i) {
            y[(ptrdiff_t)i * incy] *= beta;
        }
    }

    const int VL = (int)svcntw();
    std::vector<float> Ap; // 패널 버퍼
    std::vector<float> Xp; // x 블록 버퍼
    std::vector<float> ypart; ypart.resize((size_t)((M < VL) ? M : VL));

    for (int m0 = 0; m0 < M; m0 += VL) {
        const int MB = ((M - m0) < VL) ? (M - m0) : VL;

        // y 부분합 초기화
        std::fill(ypart.begin(), ypart.begin() + MB, 0.0f);

        for (int n0 = 0; n0 < N; n0 += VL) {
            int MB_blk = 0, NB_blk = 0;
            // 현재 (m0..m0+MB-1) x (n0..n0+NB-1) 패널 패킹
            pack_sgemv_panel_f32(A, lda, x, incx, M, N, m0, n0, MB_blk, NB_blk, Ap, Xp);

            // 커널 호출: ypart += A_panel * x_block (ZA 활용)
            sgemv_kernel_sme1_panel_f32(Ap.data(), Xp.data(), MB_blk, NB_blk, ypart.data());
        }

        // alpha 스케일 적용 후 y에 누적
        if (alpha != 1.0f) {
            for (int ib = 0; ib < MB; ++ib) ypart[ib] *= alpha;
        }
        for (int ib = 0; ib < MB; ++ib) {
            y[(ptrdiff_t)(m0 + ib) * incy] += ypart[ib];
        }
    }
}