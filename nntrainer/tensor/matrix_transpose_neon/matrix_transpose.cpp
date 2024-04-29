#include <transpose.h>

template <typename T>
void transpose_fallback(
    unsigned int M,
    unsigned int N,
    const T* src,
    unsigned int ld_src,
    T* dst,
    unsigned int ld_dst) {
  for (unsigned int j = 0; j < N; j++) {
    for (unsigned int i = 0; i < M; i++) {
      dst[i + j * ld_dst] = src[i * ld_src + j];
    }
  } 
}


