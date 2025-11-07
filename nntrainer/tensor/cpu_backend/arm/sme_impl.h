#include <stddef.h>
#include <arm_sme.h>

int hello_sme_function();

void scopy_sve(float* y, const float* x, size_t n);

void saxpy_sve(float a, const float* __restrict x, float* __restrict y, size_t n);

float sdot_sve(const float* __restrict x, const float* __restrict y, size_t n);

float sdot_sme(const float* __restrict x, const float* __restrict y, size_t n);

/*
// Error Message:
[arm64-v8a] Compile++      : nntrainer <= sme_impl.cpp
/home/sungsik/nntrainer/nntrainer/tensor/cpu_backend/arm/sme_impl.cpp:159:7: error: function declared 'float (const float *__restrict, const float *__restrict, size_t) __arm_inout("za")' (aka 'float (const float *__restrict, const float *__restrict, unsigned long) __arm_inout("za")') was previously declared 'float (const float *__restrict, const float *__restrict, size_t)' (aka 'float (const float *__restrict, const float *__restrict, unsigned long)'), which has different SME function attributes
  159 | float sdot_sme(const float* __restrict x, const float* __restrict y, size_t n)__arm_inout("za"){
      |       ^
/home/sungsik/nntrainer/nntrainer/tensor/cpu_backend/arm/sme_impl.h:12:7: note: previous declaration is here
   12 | float sdot_sme(const float* __restrict x, const float* __restrict y, size_t n);
      |       ^
1 error generated.
*/