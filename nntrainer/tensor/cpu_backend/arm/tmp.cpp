[공성식 (Sungsik)] 2025-11-07 10:13
#include <arm_sve.h>
#include <arm_sme.h>
#include <cstddef>
#include <sys/auxv.h>

#ifndef AT_HWCAP2
#define AT_HWCAP2 26
#endif
#ifndef HWCAP2_SME
#define HWCAP2_SME   (1UL<<23)
#endif
#ifndef HWCAP2_SME2
#define HWCAP2_SME2  (1UL<<24)
#endif

static inline bool has_sme()  { return (getauxval(AT_HWCAP2) & HWCAP2_SME)  != 0; }
static inline bool has_sme2() { return (getauxval(AT_HWCAP2) & HWCAP2_SME2) != 0; }

// 핵심: SME를 '사용'하지만, 최종 합은 SVE 누산에서 읽는다 (readback 불필요)
__attribute__((target("+sme2,+sme,+sve2")))
__attribute__((arm_streaming, arm_shared_za))
static float sdot_sme2_body_vg1x4(const float* __restrict x,
                                  const float* __restrict y,
                                  size_t n)
{
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
        svmla_za32_vg1x4(/*row=*/0u, /*B(4)=*/bx4, /*A(1)=*/vx);
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

// 외부 API: HWCAP 확인 + 스트리밍/ZA 컨텍스트 여닫기
float sdot_sme2_vg1x4(const float* __restrict x,
                       const float* __restrict y,
                       size_t n)
{
    // SME 미지원이면 순수 SVE 폴백
    if (!(has_sme() && has_sme2())) {
        asm volatile("smstart sm");
        // 순수 SVE sdot
        const uint32_t step = svcntw();
        size_t i = 0; svfloat32_t vacc = svdup_f32(0.0f);
        while (i < n) {
            svbool_t pg = svwhilelt_b32((uint32_t)i, (uint32_t)n);
            svfloat32_t vx = svld1_f32(pg, &x[i]);
            svfloat32_t vy = svld1_f32(pg, &y[i]);
            vacc = svmla_f32_m(pg, vacc, vx, vy);
            i += step;
        }
        float out = svaddv_f32(svptrue_b32(), vacc);
        asm volatile("smstop sm");
        return out;
    }

    // SME 경로
    asm volatile("smstart sm");
    asm volatile("smstart za");
    float out = sdot_sme2_body_vg1x4(x, y, n);
    asm volatile("smstop za");
    asm volatile("smstop sm");
    return out;
}