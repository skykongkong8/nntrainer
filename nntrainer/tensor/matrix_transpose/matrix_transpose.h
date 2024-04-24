template <typename T>
void transpose_fallback(unsigned int M, unsigned int N, const T *src,
                        unsigned int ld_src, T *dst, unsigned int ld_dst);

template <typename T>
void transpose(unsigned int M, unsigned int N, const T *src,
               unsigned int ld_src, T *dst, unsigned int ld_dst);


