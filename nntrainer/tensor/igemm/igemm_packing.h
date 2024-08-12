#include <stddef.h>

void xnn_pack_f32_conv_goki_w(
  size_t g,
  size_t nc,
  size_t ks,
  size_t kc,
  size_t nr,
  size_t kr,
  size_t sr,
  const float* k,
  const float* b,
  const void* scale,
  float* packed_weights,
  size_t extra_bytes,
  const void* params);
