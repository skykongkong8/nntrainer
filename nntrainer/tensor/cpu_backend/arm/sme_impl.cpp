#include <sme_impl.h>
// #include <sme_hwcap.h>
#include <arm_sme.h>
#include <arm_sve.h>

#include <cstdio>
#include <sys/auxv.h>
#include <cstdlib>
#include <cstring>
#include <vector>
#include <random>
#include <chrono>
#include <algorithm>
#include <thread>
#include <pthread.h>
#include <stdint.h>
#include <stddef.h>


#ifndef AT_HWCAP2
#define AT_HWCAP2 26
#endif
#ifndef HWCAP2_SME
#define HWCAP2_SME   (1UL << 23)   // 플랫폼에 따라 값이 다를 수 있음(커널 헤더 우선)
#endif

static inline bool has_sme() {
    unsigned long hw2 = getauxval(AT_HWCAP2);
    return (hw2 & HWCAP2_SME) != 0;
}

// 스트리밍 전용 함수(여기서는 ZA는 호출자에서 여닫음)
__attribute__((__arm_streaming__))
static void sme_hello_body() {
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

__attribute__((__arm_streaming__))
static void scopy_kernel_sve(const float* __restrict x, float* __restrict y, size_t n){
    size_t i = 0;
    for (; i < n;){
        svbool_t pg = svwhilelt_b32((uint32_t)i, (uint32_t)n);
        svfloat32_t vx = svld1_f32(pg, &x[i]);
        svst1_f32(pg, &y[i], vx);
        i += svcntw();
    }
}

void scopy_sve(float* y, const float* x, size_t n){
    asm volatile("smstart sm");
    scopy_kernel_sve(x, y, n);
    asm volatile("smstop sm");
}

__attribute__((__arm__streaming__))
static void saxpy_sve_kernel(float a, const float* __restrict x, float* __restrict y, size_t n){
    size_t i = 0;
    const svfloat32_t va = svdup_f32(a);
    while (i < n){
        svbool_t pg = svwhilelt_b32((uint32_t)i, (uint32_t)n);
        svfloat32_t vx = svld1_f32(pg, &x[i]);
        svfloat32_t vy = svld1_f32(pg, &y[i]);
        svfloat32_t vout = svmla_f32_m(pg, vy, vx, va); // y = a*x + y
        svst1_f32(pg, &y[i], vout);
        i += svcntw();
    }
}

void saxpy_sve(float a, const float* __restrict x, float* __restrict y, size_t n){
    asm volatile("smstart sm");
    saxpy_sve_kernel(a, x, y, n);
    asm volatile("smstop sm");
}

__attribute__ ((arm_streaming))
static float sdot_sve_kernel(const float* __restrict x, const float* __restrict y, size_t n){
    size_t i = 0;
    svfloat32_t vacc = svdup_f32(0.0F);
    while (i < n){
        svbool_t pg = svwhilelt_b32((uint32_t)i, (uint32_t)n);
        svfloat32_t vx = svld1_f32(pg, &x[i]);
        svfloat32_t vy = svld1_f32(pg, &y[i]);
        vacc = svmla_f32_m(pg, vacc, vx, vy); // acc += x * y
        i += svcntw();
    }
    return svaddv_f32(svptrue_b32(), vacc);
}

float sdot_sve(const float* __restrict x, const float* __restrict y, size_t n){
    asm volatile("smstart sm");
    float ret = sdot_sve_kernel(x, y, n);
    asm volatile("smstop sm");
    return ret;
}

__attribute__((target("+sme2,+sme,+sve2")))
__attribute__((__arm_shared_za, __arm_streaming))
static float sdot_sme2_kernel(const float* __restrict x,
                                  const float* __restrict y,
                                  size_t n) __arm_streaming
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


float sdot_sme(const float* __restrict x, const float* __restrict y, size_t n){
    if (0){
    // if (!has_sme()){
        return sdot_sve_kernel(x, y, n);
    } else{
        asm volatile("smstart sm");
        asm volatile("smstart za");
        asm volatile("zero {za}");

        float ret = sdot_sme2_kernel(x, y, n);

        asm volatile("smstop sm");
        asm volatile("smstop za");

        return ret;
    }
}
