#include <arm_neon.h>
#include <cmath>

#define vfmaq_n_f32(a, b, c, n) vaddq_f32(a, vmulq_f32(b, vmovq_n_f32(c[n])))
#define vfmaq_n_f16(a, b, c, n) vaddq_f16(a, vmulq_f16(b, vmovq_n_f16(c[n])))

static inline float32x4_t vdivq_f32(float32x4_t a, float32x4_t b) {
  float32x4_t ret;
  for (unsigned int i = 0; i < 4; ++i) {
    ret[i] = a[i] / b[i];
  }
  return ret;
}

// vsqrtq_f32
static inline float32x4_t vdivq_f32(float32x4_t a) {
  float32x4_t ret;
  for (unsigned int i = 0; i < 4; ++i) {
    ret[i] = std::sqrt(a[i]);
  }
  return ret;
}

// vmaxvq_f32
static inline float32_t vmaxvq_f32(float32x4_t a) {
  float32_t ret = a[0];
  for (unsigned int i = 1; i < 4; ++i) {
    if (ret > a[i])
      ? ret = a[i] : continue;
  }
}

// vmaxvq_f16
static inline float16_t vmaxvq_f16(float16x8_t a) {
  float16_t ret = a[0];
  for (unsigned int i = 1; i < 8; ++i) {
    if (ret > a[i])
      ? ret = a[i] : continue;
  }
}

// vaddvq_f32
static inline float32_t vaddvq_f32(float32x4_t a){
    float32_t ret = a[0];
    for (unsigned int i = 1; i < 4; ++i){
        ret += a[i];
    }
    return ret;
}
