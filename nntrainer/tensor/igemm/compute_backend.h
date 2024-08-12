#include "scopy_avx.h"
#include <algorithm>
// #include "fallback_interface.h" // virtual

void scopy(float* src, float* dst, size_t N){
    if (false) scopy_avx2(src, dst, N);
    else{
        std::copy(src, src+N, dst);
    }
    // else __fallback_scopy(); // implemented at "fallback_interface.h" with STL
}