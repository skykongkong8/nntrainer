#include <stddef.h>
#include <stdlib.h>
#include <assert.h>

static size_t divide_round_up(size_t n, size_t q) {
  return (n % q == 0) ? n / q : n / q + 1;
}

static size_t round_up(size_t n, size_t q) {
  return divide_round_up(n, q) * q;
}

template <class T> inline static T DivideRoundUp(T x, T q) {
  return x / q + T(x % q != 0);
}

template <class T> inline static T RoundUp(T x, T q) {
  return q * DivideRoundUp(x, q);
}

template <class T>
static inline T *alignedMalloc(int sz) {
  void *addr = 0;
  int iRet = posix_memalign(&addr, 64, sz * sizeof(T));
  assert(0 == iRet);
  return (T *)addr;
}

static inline size_t GetMaxCacheSize() {
//   #if XNN_ARCH_ARM || XNN_ARCH_ARM64
//     // DynamIQ max: 4 MB
//     size_t max_cache_size = 4 * 1024 * 1024;
//   #else
//     // Intel eDRAM max: 128 MB
    size_t max_cache_size = 128 * 1024 * 1024;
//   #endif
  return max_cache_size;
}

// typedef void (*igemm_ukernel_fn)(size_t mr, size_t nr, size_t kc, size_t ks,
//                                  const float **a, const float *w, float *c,
//                                  size_t cm_stride, size_t cn_stride,
//                                  size_t a_offset, const float *zero);
typedef void (igemm_ukernel_fn)(size_t, size_t, size_t, size_t,
                                 const float **, const float *, float *,
                                 size_t , size_t ,
                                 size_t , const float *);

void igemm(float *k, float *a, float *b, float* c_ret, size_t input_height,
           size_t input_width, size_t kernel_height, size_t kernel_width,
           size_t padding_height, size_t padding_width, size_t subsampling,
           size_t dilation, size_t group_input_channels,
           size_t group_output_channels, unsigned int mr, unsigned int nr,
           unsigned int kr, unsigned int sr);
