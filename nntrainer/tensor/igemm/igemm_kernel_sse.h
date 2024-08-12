#include <assert.h>
#include <stddef.h>
#include <stdint.h>

void xnn_f32_igemm_minmax_ukernel_1x8__sse_load1(
  size_t mr, size_t nc, size_t kc, size_t ks, const float **a, const float *w,
  float *c, size_t cm_stride, size_t cn_stride, size_t a_offset,
  const float *zero);
// ,const union xnn_f32_minmax_params params[restrict XNN_MIN_ELEMENTS(1)]
