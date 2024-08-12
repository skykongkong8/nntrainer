#include <immintrin.h>
#include "scopy_avx.h"

void scopy_avx2(float* src, float* dest, size_t n) {  
    __m256 vsrc;  
      
    for(; n >= 8; n -= 8) {  
        vsrc = _mm256_loadu_ps(&src[0]);  
        _mm256_storeu_ps(&dest[0], vsrc);  
          
        src += 8;  
        dest += 8;  
    }  
      
    for(; n > 0; --n) {  
        dest[0] += *src++;  
        dest += 1;  
    }  
} 
