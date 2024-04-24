
inline void transpose_kernel_4x4(const float *src, unsigned int ld_src,
                                 float *dst, unsigned int ld_dst);

inline void transpose_kernel_8x8(const float *src, unsigned int ld_src,
                                 float *dst, unsigned int ld_dst);

template <typename M>
void transpose_kernel_remain(unsigned N, const float *src, unsigned int ld_src,
                             float *dst, unsigned int ld_dst);
