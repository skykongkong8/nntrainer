#include <stddef.h>
#include <arm_sme.h>

int hello_sme_function();

void scopy_sve(float* y, const float* x, size_t n);

void saxpy_sve(float a, const float* __restrict x, float* __restrict y, size_t n);

float sdot_sve(const float* __restrict x, const float* __restrict y, size_t n);

float sdot_sme(const float* __restrict x, const float* __restrict y, size_t n);
