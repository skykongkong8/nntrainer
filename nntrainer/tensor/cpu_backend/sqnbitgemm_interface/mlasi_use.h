/*++

Copyright (c) Microsoft Corporation.  All rights reserved.

Licensed under the MIT License.

Module Name:

    mlasi.h

Abstract:

    This module contains the private data structures and procedure prototypes
    for the Microsoft Machine Learning algebra subprogram library.

--*/

#pragma once

#include <algorithm>
#include <cmath>
#include <functional>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>
#include <type_traits>

#ifdef MLAS_NO_EXCEPTION
#if defined(__ANDROID__)
#include <android/log.h>
#else
#include <iostream>
#endif
#endif  // MLAS_NO_EXCEPTION

#include "mlas.h"

#if defined(_WIN32)
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#include <intrin.h>
#else
#if defined(__arm__) || defined(__aarch64__)
#include <arm_neon.h>
#endif
#if defined(__x86_64__) || defined(__i386__)
#if !defined(signature_VORTEX_ebx) && !defined(signature_NEXGEN_ebx) && !defined(signature_AMD_ebx)//workaround for Bug 96238 - [i386] cpuid.h header needs include guards
#include <cpuid.h>
#endif
#if defined(__GNUC__) && __GNUC__ >= 12
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wmaybe-uninitialized"  // GCC 12 warns about uninitialized variables in immintrin.h.
#include <immintrin.h>
#pragma GCC diagnostic pop
#else
#include <immintrin.h>
#endif
#endif
#if defined(__VSX__)
#include <altivec.h>
// Undefine unwanted aliases from altivec.h.
#undef vector
#undef pixel
#undef bool
#endif
#if defined(__loongarch64)
#include <lsxintrin.h>
#endif
#if defined(MLAS_TARGET_WASM_SIMD)
#include <wasm_simd128.h>
#endif
#endif

//
// Macro to place variables at a specified alignment.
//

#ifdef _WIN32
#define MLAS_DECLSPEC_ALIGN(variable, alignment) DECLSPEC_ALIGN(alignment) variable
#else
#define MLAS_DECLSPEC_ALIGN(variable, alignment) variable __attribute__ ((aligned(alignment)))
#endif

//
// Macro to force inline expansion of a function.
//

#if defined(_MSC_VER)
#define inline __forceinline
#else
#define inline __attribute__ ((always_inline)) inline
#endif

//
// Macro to tag globals as internal data shared with kernels written in
// assembly. These globals are marked with having hidden visibility to avoid
// needing to access the data through the global object table.
//

#if defined(_MSC_VER)
#define MLAS_INTERNAL_DATA extern "C"
#else
#define MLAS_INTERNAL_DATA extern "C" __attribute ((visibility("hidden")))
#endif

//
// Macro to suppress unreferenced parameter warnings.
//

#define MLAS_UNREFERENCED_PARAMETER(parameter) ((void)(parameter))

#ifdef MLAS_NO_EXCEPTION

inline
void
MlasPrintFinalMessage(const std::string& msg)
{
#if defined(__ANDROID__)
    __android_log_print(ANDROID_LOG_ERROR, "mlas", "%s", msg.c_str());
#else
    // TODO, consider changing the output of the error message from std::cerr to logging when the
    // exceptions are disabled, since using std::cerr might increase binary size, and std::cerr
    // output might not be easily accesible on some systems such as mobile
    // TODO, see if we need to change the output of the error message from std::cerr to NSLog for
    // iOS
    std::cerr << msg << std::endl;
#endif
}

#define MLAS_THROW_EX(ex, what)     \
    do {                            \
        std::string msg = #ex;      \
        msg.append(what);           \
        MlasPrintFinalMessage(msg); \
        abort();                    \
    } while (false)

#else

#define MLAS_THROW_EX(ex, ...) throw ex(__VA_ARGS__)

#endif  // MLAS_NO_EXCEPTION

//
// Select the threading model.
//
// N.B. BUILD_MLAS_NO_ONNXRUNTIME is used to build MLAS test code outside
// of the ONNX Runtime source tree. OpenMP may or may not be enabled in this
// configuration.
//

#if !defined(BUILD_MLAS_NO_ONNXRUNTIME)
#include "./core/platform/threadpool.h"

#include "./core/common/cpuid_info.h"
using MLAS_CPUIDINFO = onnxruntime::CPUIDInfo;

#include "./core/framework/float16.h"

#else  // BUILD_MLAS_NO_ONNXRUNTIME

class MLASCPUIDInfo
{
   public:
    static const MLASCPUIDInfo& GetCPUIDInfo()
    {
        static MLASCPUIDInfo cpuid_info;
        return cpuid_info;
    }

    // ARM
    bool HasArmNeonDot() const { return has_arm_neon_dot_; }

    bool HasFp16VectorAcceleration() const { return has_fp16_; }

    uint32_t GetCurrentCoreIdx() const { return 0xFFFFFFFF; }

    int32_t GetCurrentUarch() const { return -1; }

    int32_t GetCoreUarch(uint32_t coreId) const { return -1; }

    bool IsCoreArmv8NarrowLd(uint32_t coreId) const { return false; }

    bool IsCurrentCoreArmv8NarrowLd() const { return false; }

    bool HasArmNeon_I8MM() const { return has_arm_neon_i8mm_; }

    bool HasArmSVE_I8MM() const { return has_arm_sve_i8mm_; }

    bool HasArmNeon_BF16() const { return has_arm_neon_bf16_; }

   private:
    MLASCPUIDInfo();

    bool has_arm_neon_dot_{false};
    bool has_fp16_{false};
    bool has_arm_neon_i8mm_{false};
    bool has_arm_sve_i8mm_{false};
    bool has_arm_neon_bf16_{false};
};
using MLAS_CPUIDINFO = MLASCPUIDInfo;

#if defined(MLAS_TARGET_ARM64)
/**
 * @brief IDs for cpu microarchitectures.
 *
 * Copied from python cpuinfo package. Can't use the definition
 * from cpuinfo directly as it causes lots of compilation issues
 * in many platforms that we support.
 */
enum MlasUArch {
    cpuinfo_uarch_unknown = 0,

    /** ARM Cortex-A32. */
    cpuinfo_uarch_cortex_a32 = 0x00300332,
    /** ARM Cortex-A35. */
    cpuinfo_uarch_cortex_a35 = 0x00300335,
    /** ARM Cortex-A53. */
    cpuinfo_uarch_cortex_a53 = 0x00300353,
    /** ARM Cortex-A55 revision 0 (restricted dual-issue capabilities compared to revision 1+). */
    cpuinfo_uarch_cortex_a55r0 = 0x00300354,
    /** ARM Cortex-A55. */
    cpuinfo_uarch_cortex_a55 = 0x00300355,
    /** ARM Cortex-A57. */
    cpuinfo_uarch_cortex_a57 = 0x00300357,
    /** ARM Cortex-A65. */
    cpuinfo_uarch_cortex_a65 = 0x00300365,
    /** ARM Cortex-A72. */
    cpuinfo_uarch_cortex_a72 = 0x00300372,
    /** ARM Cortex-A73. */
    cpuinfo_uarch_cortex_a73 = 0x00300373,
    /** ARM Cortex-A75. */
    cpuinfo_uarch_cortex_a75 = 0x00300375,
    /** ARM Cortex-A76. */
    cpuinfo_uarch_cortex_a76 = 0x00300376,
    /** ARM Cortex-A77. */
    cpuinfo_uarch_cortex_a77 = 0x00300377,
    /** ARM Cortex-A78. */
    cpuinfo_uarch_cortex_a78 = 0x00300378,
};

#endif // MLAS_TARGET_ARM64

//
// Define MLAS_FP16
//
#include "mlas_float16.h"

namespace onnxruntime
{
struct MLFloat16 {
    uint16_t val{0};

    MLFloat16() = default;
    explicit constexpr MLFloat16(uint16_t x) : val(x) {}
    explicit MLFloat16(float ff) : val(MLAS_Float2Half(ff)) {}

    float ToFloat() const { return MLAS_Half2Float(val); }

    operator float() const { return ToFloat(); }

    MLFloat16& operator=(float ff)
    {
        val = MLAS_Float2Half(ff);
        return *this;
    }
};

inline bool
operator==(const MLFloat16& left, const MLFloat16& right)
{
    return left.val == right.val;
}

inline bool
operator!=(const MLFloat16& left, const MLFloat16& right)
{
    return left.val != right.val;
}

}

#endif  // BUILD_MLAS_NO_ONNXRUNTIME

static_assert(sizeof(MLAS_FP16) == FP16_SIZE);


//
// Define the maximum number of threads supported by this implementation.
//

#define MLAS_MAXIMUM_THREAD_COUNT                   16

//
// Define the default strides to step through slices of the input matrices.
//

#define MLAS_SGEMM_STRIDEN                          128
#define MLAS_SGEMM_STRIDEK                          128
#define MLAS_SGEMM_PACKED_STRIDEN                   128
#define MLAS_SGEMM_PACKED_STRIDEK                   256
#define MLAS_DGEMM_STRIDEN                          64
#define MLAS_DGEMM_STRIDEK                          128

//
// Define the alignment for segmenting a GEMM operation across multiple
// threads.
//
// All of the SGEMM kernels can efficiently handle 16 elements. AVX512F can
// efficiently handle 32 elements, but making this value dynamic is not worth
// the effort at this time.
//

#define MLAS_SGEMM_STRIDEN_THREAD_ALIGN             16
#define MLAS_DGEMM_STRIDEN_THREAD_ALIGN             8
#define MLAS_QGEMM_STRIDEN_THREAD_ALIGN             16

//
// Define the prototypes of the platform optimized routines.
//


struct MLAS_SQNBIT_GEMM_DISPATCH;

extern const MLAS_SQNBIT_GEMM_DISPATCH MlasSQNBitGemmDispatchNeon;

extern const MLAS_SQNBIT_GEMM_DISPATCH MlasSQNBitGemmDispatchAvx2;

extern const MLAS_SQNBIT_GEMM_DISPATCH MlasSQNBitGemmDispatchAvx2vnni;

extern const MLAS_SQNBIT_GEMM_DISPATCH MlasSQNBitGemmDispatchAvx512;

extern const MLAS_SQNBIT_GEMM_DISPATCH MlasSQNBitGemmDispatchAvx512vnni;

typedef
void(MLASCALL MLAS_CAST_F16_TO_F32_KERNEL)(
    const unsigned short* Source,
    float* Destination,
    size_t Count
);

typedef void(MLASCALL MLAS_CAST_F32_TO_F16_KERNEL)(
    const float* Source,
    unsigned short* Destination,
    size_t Count
);

struct MLAS_PLATFORM {

    MLAS_PLATFORM(void);

#if defined(MLAS_TARGET_AMD64)
    uint32_t NchwcBlockSize;
    uint32_t PreferredBufferAlignment;
    int32_t MaximumThreadCount;
#elif defined(MLAS_TARGET_ARM64)
    static constexpr int32_t MaximumThreadCount = MLAS_MAXIMUM_THREAD_COUNT * 4;
#else
    static constexpr int32_t MaximumThreadCount = MLAS_MAXIMUM_THREAD_COUNT;
#endif
    const MLAS_SQNBIT_GEMM_DISPATCH* SQNBitGemmDispatch{nullptr};

    MLAS_CAST_F16_TO_F32_KERNEL* CastF16ToF32Kernel;
    MLAS_CAST_F32_TO_F16_KERNEL* CastF32ToF16Kernel;
};
inline
MLAS_PLATFORM& GetMlasPlatform(){
    static MLAS_PLATFORM MlasPlatform;
    return MlasPlatform;
}

//
// Threading support.
//

typedef
void
(MLAS_THREADED_ROUTINE)(
    void* Context,
    ptrdiff_t Index
    );

void
MlasExecuteThreaded(
    MLAS_THREADED_ROUTINE* ThreadedRoutine,
    void* Context,
    ptrdiff_t Iterations,
    MLAS_THREADPOOL* ThreadPool
    );

constexpr
size_t
MlasDivRoundup(size_t up, size_t down)
{
    return (up + down - 1) / down;
}

/**
 * @brief Distribute multiple iterations of work over a thread pool if supported
 *
 * @param ThreadPool [IN]          Optional thread pool. Ignored when using OpenMP
 * @param Iterations [IN]          Total number of iterations
 * @param Work [IN]                Logic for computing a range of iterations [begin, end)
 */
void
MlasTrySimpleParallel(
    MLAS_THREADPOOL* ThreadPool,
    const std::ptrdiff_t Iterations,
    const std::function<void(std::ptrdiff_t tid)>& Work
    );


/**
 * @brief Distribute many iterations of work over a thread pool if supported.
 * This function is for small workloads in non-performance critical situation.
 *
 * @param ThreadPool [IN]          Optional thread pool. Ignored when using OpenMP
 * @param Iterations [IN]          Total number of iterations
 * @param Work [IN]                Logic for computing a range of iterations [begin, end)
 */
void
MlasTryBatchParallel(
	MLAS_THREADPOOL * ThreadPool,
	const std::ptrdiff_t Iterations,
	const std::function<void(std::ptrdiff_t tid)>& Work
    );


inline
ptrdiff_t
MlasGetMaximumThreadCount(
    MLAS_THREADPOOL* ThreadPool
    )
{
    
// // #if defined(BUILD_MLAS_NO_ONNXRUNTIME)
#if 1
    MLAS_UNREFERENCED_PARAMETER(ThreadPool);
    return 1;
#else
    return onnxruntime::concurrency::ThreadPool::DegreeOfParallelism(ThreadPool);
#endif
}

inline
void
MlasPartitionWork(
    ptrdiff_t ThreadId,
    ptrdiff_t ThreadCount,
    size_t TotalWork,
    size_t* WorkIndex,
    size_t* WorkRemaining
    )
{
    const size_t WorkPerThread = TotalWork / ThreadCount;
    const size_t WorkPerThreadExtra = TotalWork % ThreadCount;

    if (size_t(ThreadId) < WorkPerThreadExtra) {
        *WorkIndex = (WorkPerThread + 1) * ThreadId;
        *WorkRemaining = WorkPerThread + 1;
    } else {
        *WorkIndex = WorkPerThread * ThreadId + WorkPerThreadExtra;
        *WorkRemaining = WorkPerThread;
    }
}

//
// Define the minimum floating point value (and its bit value equivalent) that
// has no fractional bits. This number can be used for fast rounding of floating
// point numbers to integers.
//

#define MLAS_ROUNDING_BIAS_MAGIC                    12582912.f
#define MLAS_ROUNDING_BIAS_MAGIC_BITS               0x4B400000

//
// Helpers to cast a floating point type to and from an integer bit format.
//
#if defined(_MSC_VER) && !defined(__clang__)
  #pragma warning(push)
  // VC++ suggests we can attempt to make 'MlasBitsOfFp32' constexpr, but it is not valid.
  #pragma warning(disable:26497)
#endif

inline
uint32_t
MlasBitsOfFp32(
    float FloatValue
    )
{
    union {
        uint32_t IntegerValue;
        float FloatValue;
    } u;
    u.FloatValue = FloatValue;
    return u.IntegerValue;
}

inline
float
MlasFp32FromBits(
    uint32_t IntegerValue
    )
{
    union {
        uint32_t IntegerValue;
        float FloatValue;
    } u;
    u.IntegerValue = IntegerValue;
    return u.FloatValue;
}


#if defined(_M_ARM64)
#ifndef vmaxvq_f32
#define vmaxvq_f32(src) neon_fmaxv(src)
#endif
#ifndef vminvq_f32
#define vminvq_f32(src) neon_fminv(src)
#endif
#endif

//
// Cross-platform wrappers for 32-bit vector intrinsics.
//

#if defined(MLAS_TARGET_ARM)
#define MLAS_NEON_INTRINSICS
#define MLAS_NEON32_INTRINSICS
#elif defined(MLAS_TARGET_ARM64) || defined(MLAS_TARGET_ARM64EC)
#define MLAS_NEON_INTRINSICS
#define MLAS_NEON64_INTRINSICS
#elif defined(MLAS_TARGET_POWER)
#define MLAS_VSX_INTRINSICS
#elif defined(MLAS_TARGET_AMD64_IX86)
#define MLAS_SSE2_INTRINSICS
#if defined(__SSE4_1__) || (defined(_MSC_VER) && defined(__AVX__))
#define MLAS_SSE41_INTRINSICS
#endif
#if defined(__AVX__)
#define MLAS_AVX_INTRINSICS
#endif
#if defined(__AVX2__)
#define MLAS_AVX2_INTRINSICS
#endif
#if defined(__FMA__) || (defined(_MSC_VER) && defined(__AVX2__))
#define MLAS_FMA3_INTRINSICS
#endif
#elif defined(MLAS_TARGET_WASM_SIMD)
#define MLAS_WASM_SIMD_INTRINSICS
#elif defined(MLAS_TARGET_LARCH64)
#define MLAS_LSX_INTRINSICS
#endif

#if defined(MLAS_NEON_INTRINSICS)
typedef float32x4_t MLAS_FLOAT32X4;
typedef int32x4_t MLAS_INT32X4;
#elif defined(MLAS_SSE2_INTRINSICS)
typedef __m128 MLAS_FLOAT32X4;
typedef __m128i MLAS_INT32X4;
#elif defined(MLAS_VSX_INTRINSICS)
typedef __vector float MLAS_FLOAT32X4;
typedef __vector int MLAS_INT32X4;
typedef __vector unsigned MLAS_UINT32X4;
#elif defined(MLAS_WASM_SIMD_INTRINSICS)
typedef v128_t MLAS_FLOAT32X4;
typedef v128_t MLAS_INT32X4;
#elif defined(MLAS_LSX_INTRINSICS)
typedef __m128 MLAS_FLOAT32X4;
typedef __m128i MLAS_INT32X4;
#else
typedef float MLAS_FLOAT32X4 __attribute__ ((vector_size(16)));
typedef int32_t MLAS_INT32X4 __attribute__ ((vector_size(16)));
#endif

inline
MLAS_INT32X4
MlasReinterpretAsInt32x4(MLAS_FLOAT32X4 Vector)
{
#if defined(MLAS_NEON_INTRINSICS)
    return vreinterpretq_s32_f32(Vector);
#elif defined(MLAS_SSE2_INTRINSICS)
    return _mm_castps_si128(Vector);
#elif defined(MLAS_LSX_INTRINSICS)
    return (MLAS_INT32X4)Vector;
#else
    return MLAS_INT32X4(Vector);
#endif
}

inline
MLAS_INT32X4
MlasCastToInt32x4(MLAS_FLOAT32X4 Vector)
{
#if defined(MLAS_NEON_INTRINSICS)
    return vcvtq_s32_f32(Vector);
#elif defined(MLAS_SSE2_INTRINSICS)
    return _mm_cvttps_epi32(Vector);
#elif defined(MLAS_VSX_INTRINSICS)
    return vec_cts(Vector, 0);
#elif defined(MLAS_LSX_INTRINSICS)
    return __lsx_vftint_w_s(Vector);
#elif defined(MLAS_WASM_SIMD_INTRINSICS)
    return (MLAS_INT32X4)__builtin_convertvector((__f32x4)Vector, __i32x4);
#else
    return MLAS_INT32X4{int32_t(Vector[0]), int32_t(Vector[1]), int32_t(Vector[2]), int32_t(Vector[3])};
#endif
}

inline
MLAS_FLOAT32X4
MlasCastToFloat32x4(MLAS_INT32X4 Vector)
{
#if defined(MLAS_NEON_INTRINSICS)
    return vcvtq_f32_s32(Vector);
#elif defined(MLAS_SSE2_INTRINSICS)
    return _mm_cvtepi32_ps(Vector);
#elif defined(MLAS_VSX_INTRINSICS)
    return vec_ctf(Vector, 0);
#elif defined(MLAS_WASM_SIMD_INTRINSICS)
    return wasm_f32x4_convert_i32x4(Vector);
#elif defined(MLAS_LSX_INTRINSICS)
    return __lsx_vffint_s_w(Vector);
#else
    return MLAS_FLOAT32X4{float(Vector[0]), float(Vector[1]), float(Vector[2]), float(Vector[3])};
#endif
}

inline
MLAS_INT32X4
MlasBroadcastInt32x4(int32_t Value)
{
#if defined(MLAS_NEON_INTRINSICS)
    return vdupq_n_s32(Value);
#elif defined(MLAS_SSE2_INTRINSICS)
    return _mm_set1_epi32(Value);
#elif defined(MLAS_WASM_SIMD_INTRINSICS)
    return wasm_i32x4_splat(Value);
#elif defined(MLAS_VSX_INTRINSICS)
    return vec_splats(Value);
#elif defined(MLAS_LSX_INTRINSICS)
    return __lsx_vreplgr2vr_w(Value);
#else
    return MLAS_INT32X4{Value, Value, Value, Value};
#endif
}

inline
MLAS_INT32X4
MlasLoadInt32x4(const int32_t* Buffer)
{
#if defined(MLAS_NEON_INTRINSICS)
    return vld1q_s32(Buffer);
#elif defined(MLAS_SSE2_INTRINSICS)
    return _mm_loadu_si128((const __m128i*)Buffer);
#elif defined(MLAS_VSX_INTRINSICS)
    return vec_vsx_ld(0, Buffer);
#elif defined(MLAS_WASM_SIMD_INTRINSICS)
    return wasm_v128_load(Buffer);
#elif defined(MLAS_LSX_INTRINSICS)
    return __lsx_vld((const MLAS_INT32X4*)Buffer, 0);
#else
    return *((MLAS_INT32X4*)Buffer);
#endif
}

inline
void
MlasStoreInt32x4(int32_t* Buffer, MLAS_INT32X4 Vector)
{
#if defined(MLAS_NEON_INTRINSICS)
    vst1q_s32(Buffer, Vector);
#elif defined(MLAS_SSE2_INTRINSICS)
    _mm_storeu_si128((__m128i*)Buffer, Vector);
#elif defined(MLAS_VSX_INTRINSICS)
    vec_vsx_st(Vector, 0, Buffer);
#elif defined(MLAS_WASM_SIMD_INTRINSICS)
    wasm_v128_store(Buffer, Vector);
#elif defined(MLAS_LSX_INTRINSICS)
    __lsx_vst(Vector, (MLAS_INT32X4 *)Buffer, 0);
#else
    *((MLAS_INT32X4*)Buffer) = Vector;
#endif
}

inline
MLAS_INT32X4
MlasAddInt32x4(MLAS_INT32X4 Vector1, MLAS_INT32X4 Vector2)
{
#if defined(MLAS_NEON_INTRINSICS)
    return vaddq_s32(Vector1, Vector2);
#elif defined(MLAS_SSE2_INTRINSICS)
    return _mm_add_epi32(Vector1, Vector2);
#elif defined(MLAS_WASM_SIMD_INTRINSICS)
    return wasm_i32x4_add(Vector1, Vector2);
#elif defined(MLAS_VSX_INTRINSICS)
    return vec_add(Vector1, Vector2);
#elif defined(MLAS_LSX_INTRINSICS)
    return __lsx_vadd_w(Vector1, Vector2);
#else
    return Vector1 + Vector2;
#endif
}

inline
MLAS_INT32X4
MlasSubtractInt32x4(MLAS_INT32X4 Vector1, MLAS_INT32X4 Vector2)
{
#if defined(MLAS_NEON_INTRINSICS)
    return vsubq_s32(Vector1, Vector2);
#elif defined(MLAS_SSE2_INTRINSICS)
    return _mm_sub_epi32(Vector1, Vector2);
#elif defined(MLAS_WASM_SIMD_INTRINSICS)
    return wasm_i32x4_sub(Vector1, Vector2);
#elif defined(MLAS_LSX_INTRINSICS)
    return __lsx_vsub_w(Vector1, Vector2);
#else
    return Vector1 - Vector2;
#endif
}

inline
MLAS_INT32X4
MlasAndInt32x4(MLAS_INT32X4 Vector1, MLAS_INT32X4 Vector2)
{
#if defined(MLAS_NEON_INTRINSICS)
    return vandq_s32(Vector1, Vector2);
#elif defined(MLAS_SSE2_INTRINSICS)
    return _mm_and_si128(Vector1, Vector2);
#elif defined(MLAS_WASM_SIMD_INTRINSICS)
    return wasm_v128_and(Vector1, Vector2);
#elif defined(MLAS_LSX_INTRINSICS)
    return __lsx_vand_v(Vector1, Vector2);
#else
    return Vector1 & Vector2;
#endif
}

inline
MLAS_INT32X4
MlasOrInt32x4(MLAS_INT32X4 Vector1, MLAS_INT32X4 Vector2)
{
#if defined(MLAS_NEON_INTRINSICS)
    return vorrq_s32(Vector1, Vector2);
#elif defined(MLAS_SSE2_INTRINSICS)
    return _mm_or_si128(Vector1, Vector2);
#elif defined(MLAS_WASM_SIMD_INTRINSICS)
    return wasm_v128_or(Vector1, Vector2);
#elif defined(MLAS_LSX_INTRINSICS)
    return __lsx_vor_v(Vector1, Vector2);
#else
    return Vector1 | Vector2;
#endif
}

inline
MLAS_INT32X4
MlasAndNotInt32x4(MLAS_INT32X4 VectorNot, MLAS_INT32X4 Vector)
{
#if defined(MLAS_NEON_INTRINSICS)
    return vandq_s32(vmvnq_s32(VectorNot), Vector);
#elif defined(MLAS_SSE2_INTRINSICS)
    return _mm_andnot_si128(VectorNot, Vector);
#elif defined(MLAS_WASM_SIMD_INTRINSICS)
    return wasm_v128_andnot(Vector, VectorNot);
#elif defined(MLAS_LSX_INTRINSICS)
    return __lsx_vandn_v(VectorNot, Vector);
#else
    return (~VectorNot) & Vector;
#endif
}

inline
MLAS_INT32X4
MlasXorInt32x4(MLAS_INT32X4 Vector1, MLAS_INT32X4 Vector2)
{
#if defined(MLAS_NEON_INTRINSICS)
    return veorq_s32(Vector1, Vector2);
#elif defined(MLAS_SSE2_INTRINSICS)
    return _mm_xor_si128(Vector1, Vector2);
#elif defined(MLAS_WASM_SIMD_INTRINSICS)
    return wasm_v128_xor(Vector1, Vector2);
#elif defined(MLAS_VSX_INTRINSICS)
    return vec_xor(Vector1, Vector2);
#elif defined(MLAS_LSX_INTRINSICS)
    return __lsx_vxor_v(Vector1, Vector2);
#else
    return Vector1 ^ Vector2;
#endif
}

inline
MLAS_INT32X4
MlasBlendInt32x4(MLAS_INT32X4 Vector1, MLAS_INT32X4 Vector2, MLAS_INT32X4 Selection)
{
    return MlasOrInt32x4(MlasAndInt32x4(Vector2, Selection), MlasAndNotInt32x4(Selection, Vector1));
}

template<unsigned ShiftCount>
inline
MLAS_INT32X4
MlasShiftLeftInt32x4(MLAS_INT32X4 Vector)
{
#if defined(MLAS_NEON_INTRINSICS)
    return vshlq_n_s32(Vector, ShiftCount);
#elif defined(MLAS_SSE2_INTRINSICS)
    return _mm_slli_epi32(Vector, ShiftCount);
#elif defined(MLAS_WASM_SIMD_INTRINSICS)
    return wasm_i32x4_shl(Vector, ShiftCount);
#elif defined(MLAS_LSX_INTRINSICS)
    return __lsx_vslli_w(Vector, ShiftCount);
#else
    return Vector << ShiftCount;
#endif
}

inline
MLAS_INT32X4
MlasMaximumInt32x4(MLAS_INT32X4 Vector1, MLAS_INT32X4 Vector2)
{
#if defined(MLAS_NEON_INTRINSICS)
    return vmaxq_s32(Vector1, Vector2);
#elif defined(MLAS_SSE41_INTRINSICS)
    return _mm_max_epi32(Vector1, Vector2);
#elif defined(MLAS_SSE2_INTRINSICS)
    return MlasBlendInt32x4(Vector2, Vector1, _mm_cmpgt_epi32(Vector1, Vector2));
#elif defined(MLAS_VSX_INTRINSICS)
    return vec_vmaxsw(Vector1, Vector2);
#elif defined(MLAS_WASM_SIMD_INTRINSICS)
    return wasm_i32x4_max(Vector1, Vector2);
#elif defined(MLAS_LSX_INTRINSICS)
    return __lsx_vmax_w(Vector1, Vector2);
#else
    return MlasBlendInt32x4(Vector2, Vector1, Vector1 > Vector2);
#endif
}

inline
MLAS_INT32X4
MlasMinimumInt32x4(MLAS_INT32X4 Vector1, MLAS_INT32X4 Vector2)
{
#if defined(MLAS_NEON_INTRINSICS)
    return vminq_s32(Vector1, Vector2);
#elif defined(MLAS_SSE41_INTRINSICS)
    return _mm_min_epi32(Vector1, Vector2);
#elif defined(MLAS_SSE2_INTRINSICS)
    return MlasBlendInt32x4(Vector2, Vector1, _mm_cmpgt_epi32(Vector2, Vector1));
#elif defined(MLAS_VSX_INTRINSICS)
    return vec_vminsw(Vector1, Vector2);
#elif defined(MLAS_WASM_SIMD_INTRINSICS)
    return wasm_i32x4_min(Vector1, Vector2);
#elif defined(MLAS_LSX_INTRINSICS)
    return __lsx_vmin_w(Vector1, Vector2);
#else
    return MlasBlendInt32x4(Vector2, Vector1, Vector2 > Vector1);
#endif
}

inline
MLAS_FLOAT32X4
MlasReinterpretAsFloat32x4(MLAS_INT32X4 Vector)
{
#if defined(MLAS_NEON_INTRINSICS)
    return vreinterpretq_f32_s32(Vector);
#elif defined(MLAS_SSE2_INTRINSICS)
    return _mm_castsi128_ps(Vector);
#elif defined(MLAS_LSX_INTRINSICS)
    return MLAS_FLOAT32X4(Vector);
#else
    return MLAS_FLOAT32X4(Vector);
#endif
}

inline
MLAS_FLOAT32X4
MlasBroadcastFloat32x4(float Value)
{
#if defined(MLAS_NEON_INTRINSICS)
    return vdupq_n_f32(Value);
#elif defined(MLAS_SSE2_INTRINSICS)
    return _mm_set1_ps(Value);
#elif defined(MLAS_WASM_SIMD_INTRINSICS)
    return wasm_f32x4_splat(Value);
#elif defined(MLAS_VSX_INTRINSICS)
    // Suppress wrong GCC warnings
    MLAS_UNREFERENCED_PARAMETER(Value);
    return vec_splats(Value);
#elif defined(MLAS_LSX_INTRINSICS)
    return MLAS_FLOAT32X4{Value, Value, Value, Value};
#else
    return MLAS_FLOAT32X4{Value, Value, Value, Value};
#endif
}

inline
MLAS_FLOAT32X4
MlasBroadcastFloat32x4(const float* Value)
{
#if defined(MLAS_NEON_INTRINSICS)
    return vld1q_dup_f32(Value);
#elif defined(MLAS_SSE2_INTRINSICS)
    return _mm_load_ps1(Value);
#elif defined(MLAS_WASM_SIMD_INTRINSICS)
    return wasm_v128_load32_splat(Value);
#elif defined(MLAS_VSX_INTRINSICS)
    return vec_splats(*Value);
#elif defined(MLAS_LSX_INTRINSICS)
    return MLAS_FLOAT32X4{*Value, *Value, *Value, *Value};
#else
    return MLAS_FLOAT32X4{*Value, *Value, *Value, *Value};
#endif
}

inline
MLAS_FLOAT32X4
MlasZeroFloat32x4(void)
{
#if defined(MLAS_NEON_INTRINSICS)
    return vdupq_n_f32(0.0f);
#elif defined(MLAS_SSE2_INTRINSICS)
    return _mm_setzero_ps();
#elif defined(MLAS_WASM_SIMD_INTRINSICS)
    return wasm_f32x4_const(0.0f, 0.0f, 0.0f, 0.0f);
#elif defined(MLAS_LSX_INTRINSICS)
    return MlasBroadcastFloat32x4(0.0f);
#else
    return MlasBroadcastFloat32x4(0.0f);
#endif
}

inline
MLAS_FLOAT32X4
MlasLoadFloat32x4(const float* Buffer)
{
#if defined(MLAS_NEON_INTRINSICS)
    return vld1q_f32(Buffer);
#elif defined(MLAS_SSE2_INTRINSICS)
    return _mm_loadu_ps(Buffer);
#elif defined(MLAS_VSX_INTRINSICS)
    return vec_vsx_ld(0, Buffer);
#elif defined(MLAS_WASM_SIMD_INTRINSICS)
    return wasm_v128_load(Buffer);
#elif defined(MLAS_LSX_INTRINSICS)
    // return MlasReinterpretAsFloat32x4(__lsx_vld((const MLAS_INT32X4 *)Buffer, 0));
    return (MLAS_FLOAT32X4)__lsx_vld((const MLAS_INT32X4 *)Buffer, 0);
#else
    return *((MLAS_FLOAT32X4*)Buffer);
#endif
}

inline
void
MlasStoreFloat32x4(float* Buffer, MLAS_FLOAT32X4 Vector)
{
#if defined(MLAS_NEON_INTRINSICS)
    vst1q_f32(Buffer, Vector);
#elif defined(MLAS_SSE2_INTRINSICS)
    _mm_storeu_ps(Buffer, Vector);
#elif defined(MLAS_VSX_INTRINSICS)
    vec_vsx_st(Vector, 0, Buffer);
#elif defined(MLAS_WASM_SIMD_INTRINSICS)
    wasm_v128_store(Buffer, Vector);
#elif defined(MLAS_LSX_INTRINSICS)
    __lsx_vst(MlasReinterpretAsInt32x4(Vector), Buffer, 0);
#else
    *((MLAS_FLOAT32X4*)Buffer) = Vector;
#endif
}

inline
void
MlasStoreAlignedFloat32x4(float* Buffer, MLAS_FLOAT32X4 Vector)
{
#if defined(MLAS_NEON_INTRINSICS)
    vst1q_f32(Buffer, Vector);
#elif defined(MLAS_SSE2_INTRINSICS)
    _mm_store_ps(Buffer, Vector);
#elif defined(MLAS_VSX_INTRINSICS)
    // Workaround for bad GCC warning that these parameters are set but not used.
    MLAS_UNREFERENCED_PARAMETER(Buffer);
    MLAS_UNREFERENCED_PARAMETER(Vector);
    vec_st(Vector, 0, Buffer);
#elif defined(MLAS_WASM_SIMD_INTRINSICS)
    wasm_v128_store(Buffer, Vector);
#elif defined(MLAS_LSX_INTRINSICS)
    MlasStoreFloat32x4(Buffer, Vector);
#else
    MlasStoreFloat32x4(Buffer, Vector);
#endif
}

template<unsigned Lane>
inline
void
MlasStoreLaneFloat32x4(float* Buffer, MLAS_FLOAT32X4 Vector)
{
#if defined(MLAS_NEON_INTRINSICS)
    vst1q_lane_f32(Buffer, Vector, Lane);
#elif defined(MLAS_SSE2_INTRINSICS)
    // N.B. When building with AVX instructions, compilers optimize the following
    // to a single vextractps instruction.
    _mm_store_ss(Buffer, _mm_shuffle_ps(Vector, Vector, _MM_SHUFFLE(Lane, Lane, Lane, Lane)));
#elif defined(MLAS_WASM_SIMD_INTRINSICS)
    *Buffer = ((__f32x4)(Vector))[Lane];
#elif defined(MLAS_LSX_INTRINSICS)
    *Buffer = Vector[Lane];
#else
    *Buffer = Vector[Lane];
#endif
}

inline
void
MlasStoreLowHalfFloat32x4(float* Buffer, MLAS_FLOAT32X4 Vector)
{
#if defined(MLAS_NEON_INTRINSICS)
    vst1_f32(Buffer, vget_low_f32(Vector));
#elif defined(MLAS_SSE2_INTRINSICS)
    _mm_storel_pi((__m64*)Buffer, Vector);
#elif defined(MLAS_VSX_INTRINSICS)
    *((long long*)Buffer) = ((__vector long long)Vector)[0];
#elif defined(MLAS_LSX_INTRINSICS)
    MlasStoreLaneFloat32x4<0>(&Buffer[0], Vector);
    MlasStoreLaneFloat32x4<1>(&Buffer[1], Vector);
#else
    MlasStoreLaneFloat32x4<0>(&Buffer[0], Vector);
    MlasStoreLaneFloat32x4<1>(&Buffer[1], Vector);
#endif
}

template<unsigned Lane>
inline
float
MlasExtractLaneFloat32x4(MLAS_FLOAT32X4 Vector)
{
#if defined(MLAS_NEON_INTRINSICS)
    return vgetq_lane_f32(Vector, Lane);
#elif defined(MLAS_SSE2_INTRINSICS)
    return _mm_cvtss_f32(_mm_shuffle_ps(Vector, Vector, _MM_SHUFFLE(Lane, Lane, Lane, Lane)));
#elif defined(MLAS_WASM_SIMD_INTRINSICS)
    return wasm_f32x4_extract_lane(Vector, Lane);
#elif defined(MLAS_LSX_INTRINSICS)
    return Vector[Lane];
#else
    return Vector[Lane];
#endif
}

#if defined(MLAS_SSE2_INTRINSICS)

template<>
inline
void
MlasStoreLaneFloat32x4<0>(float* Buffer, MLAS_FLOAT32X4 Vector)
{
    _mm_store_ss(Buffer, Vector);
}

template<>
inline
float
MlasExtractLaneFloat32x4<0>(MLAS_FLOAT32X4 Vector)
{
    return _mm_cvtss_f32(Vector);
}

template<unsigned Index0, unsigned Index1, unsigned Index2, unsigned Index3>
inline
MLAS_FLOAT32X4
MlasShuffleFloat32x4(MLAS_FLOAT32X4 Vector)
{
    return _mm_shuffle_ps(Vector, Vector, _MM_SHUFFLE(Index3, Index2, Index1, Index0));
}

#endif

#if !defined(MLAS_SSE2_INTRINSICS) && !defined(_MSC_VER)

template<unsigned Index0, unsigned Index1, unsigned Index2, unsigned Index3>
inline
MLAS_FLOAT32X4
MlasShuffleFloat32x4(MLAS_FLOAT32X4 Vector1, MLAS_FLOAT32X4 Vector2)
{
#if defined(MLAS_WASM_SIMD_INTRINSICS)
    return wasm_i32x4_shuffle(Vector1, Vector2, Index0, Index1, Index2, Index3);
#elif defined(__clang__)
    return __builtin_shufflevector(Vector1, Vector2, Index0, Index1, Index2, Index3);
#elif defined(MLAS_LSX_INTRINSICS)
    typedef int32_t GEN_INT32X4 __attribute__ ((vector_size(16)));
    return __builtin_shuffle(Vector1, Vector2, GEN_INT32X4{Index0, Index1, Index2, Index3});
#else
    return __builtin_shuffle(Vector1, Vector2, MLAS_INT32X4{Index0, Index1, Index2, Index3});
#endif
}

template<unsigned Index0, unsigned Index1, unsigned Index2, unsigned Index3>
inline
MLAS_FLOAT32X4
MlasShuffleFloat32x4(MLAS_FLOAT32X4 Vector)
{
    return MlasShuffleFloat32x4<Index0, Index1, Index2, Index3>(Vector, Vector);
}

#endif

inline
MLAS_FLOAT32X4
MlasInterleaveLowFloat32x4(MLAS_FLOAT32X4 Vector1, MLAS_FLOAT32X4 Vector2)
{
#if defined(MLAS_NEON64_INTRINSICS)
    return vzip1q_f32(Vector1, Vector2);
#elif defined(MLAS_NEON32_INTRINSICS)
    float32x4x2_t zipped = vzipq_f32(Vector1, Vector2);
    return zipped.val[0];
#elif defined(MLAS_SSE2_INTRINSICS)
    return _mm_unpacklo_ps(Vector1, Vector2);
#elif defined(MLAS_VSX_INTRINSICS)
    return vec_mergeh(Vector1, Vector2);
#elif defined(MLAS_LSX_INTRINSICS)
    return (MLAS_FLOAT32X4)__lsx_vilvl_w(MlasReinterpretAsInt32x4(Vector2), MlasReinterpretAsInt32x4(Vector1));
#else
    return MlasShuffleFloat32x4<0, 4, 1, 5>(Vector1, Vector2);
#endif
}

inline
MLAS_FLOAT32X4
MlasInterleaveHighFloat32x4(MLAS_FLOAT32X4 Vector1, MLAS_FLOAT32X4 Vector2)
{
#if defined(MLAS_NEON64_INTRINSICS)
    return vzip2q_f32(Vector1, Vector2);
#elif defined(MLAS_NEON32_INTRINSICS)
    float32x4x2_t zipped = vzipq_f32(Vector1, Vector2);
    return zipped.val[1];
#elif defined(MLAS_SSE2_INTRINSICS)
    return _mm_unpackhi_ps(Vector1, Vector2);
#elif defined(MLAS_VSX_INTRINSICS)
    return vec_mergel(Vector1, Vector2);
#elif defined(MLAS_LSX_INTRINSICS)
    return (MLAS_FLOAT32X4)__lsx_vilvh_w(MlasReinterpretAsInt32x4(Vector2), MlasReinterpretAsInt32x4(Vector1));
#else
    return MlasShuffleFloat32x4<2, 6, 3, 7>(Vector1, Vector2);
#endif
}

inline
MLAS_FLOAT32X4
MlasAddFloat32x4(MLAS_FLOAT32X4 Vector1, MLAS_FLOAT32X4 Vector2)
{
#if defined(MLAS_NEON_INTRINSICS)
    return vaddq_f32(Vector1, Vector2);
#elif defined(MLAS_SSE2_INTRINSICS)
    return _mm_add_ps(Vector1, Vector2);
#elif defined(MLAS_WASM_SIMD_INTRINSICS)
    return wasm_f32x4_add(Vector1, Vector2);
#elif defined(MLAS_VSX_INTRINSICS)
    return vec_add(Vector1, Vector2);
#elif defined(MLAS_LSX_INTRINSICS)
    return __lsx_vfadd_s(Vector1, Vector2);
#else
    return Vector1 + Vector2;
#endif
}

inline
MLAS_FLOAT32X4
MlasSubtractFloat32x4(MLAS_FLOAT32X4 Vector1, MLAS_FLOAT32X4 Vector2)
{
#if defined(MLAS_NEON_INTRINSICS)
    return vsubq_f32(Vector1, Vector2);
#elif defined(MLAS_SSE2_INTRINSICS)
    return _mm_sub_ps(Vector1, Vector2);
#elif defined(MLAS_WASM_SIMD_INTRINSICS)
    return wasm_f32x4_sub(Vector1, Vector2);
#elif defined(MLAS_VSX_INTRINSICS)
    return vec_sub(Vector1, Vector2);
#elif defined(MLAS_LSX_INTRINSICS)
    return __lsx_vfsub_s(Vector1, Vector2);
#else
    return Vector1 - Vector2;
#endif
}

inline
MLAS_FLOAT32X4
MlasMultiplyFloat32x4(MLAS_FLOAT32X4 Vector1, MLAS_FLOAT32X4 Vector2)
{
#if defined(MLAS_NEON_INTRINSICS)
    return vmulq_f32(Vector1, Vector2);
#elif defined(MLAS_SSE2_INTRINSICS)
    return _mm_mul_ps(Vector1, Vector2);
#elif defined(MLAS_WASM_SIMD_INTRINSICS)
    return wasm_f32x4_mul(Vector1, Vector2);
#elif defined(MLAS_VSX_INTRINSICS)
    // Suppress wrong GCC warnings
    MLAS_UNREFERENCED_PARAMETER(Vector1);
    MLAS_UNREFERENCED_PARAMETER(Vector2);
    return vec_mul(Vector1, Vector2);
#elif defined(MLAS_LSX_INTRINSICS)
    return __lsx_vfmul_s(Vector1, Vector2);
#else
    return Vector1 * Vector2;
#endif
}

inline
MLAS_FLOAT32X4
MlasMultiplyAddFloat32x4(MLAS_FLOAT32X4 Vector1, MLAS_FLOAT32X4 Vector2, MLAS_FLOAT32X4 Vector3)
{
#if defined(MLAS_NEON_INTRINSICS)
    return vmlaq_f32(Vector3, Vector1, Vector2);
#elif defined(MLAS_FMA3_INTRINSICS)
    return _mm_fmadd_ps(Vector1, Vector2, Vector3);
#elif defined(MLAS_SSE2_INTRINSICS)
    return _mm_add_ps(_mm_mul_ps(Vector1, Vector2), Vector3);
#elif defined(MLAS_VSX_INTRINSICS)
    return vec_madd(Vector1, Vector2, Vector3);
#elif defined(MLAS_WASM_SIMD_INTRINSICS)
    return wasm_f32x4_add(wasm_f32x4_mul(Vector1, Vector2), Vector3);
#elif defined(MLAS_LSX_INTRINSICS)
    return __lsx_vfmadd_s(Vector1, Vector2, Vector3);
#else
    return Vector1 * Vector2 + Vector3;
#endif
}

inline
MLAS_FLOAT32X4
MlasMultiplyAddFloat32x4(MLAS_FLOAT32X4 Vector1, float Scalar2, MLAS_FLOAT32X4 Vector3)
{
    return MlasMultiplyAddFloat32x4(Vector1, MlasBroadcastFloat32x4(Scalar2), Vector3);
}

inline
MLAS_FLOAT32X4
MlasMultiplyAddFloat32x4(MLAS_FLOAT32X4 Vector1, MLAS_FLOAT32X4 Vector2, float Scalar3)
{
    return MlasMultiplyAddFloat32x4(Vector1, Vector2, MlasBroadcastFloat32x4(Scalar3));
}

inline
MLAS_FLOAT32X4
MlasDivideFloat32x4(MLAS_FLOAT32X4 Vector1, MLAS_FLOAT32X4 Vector2)
{
#if defined(MLAS_NEON64_INTRINSICS)
    return vdivq_f32(Vector1, Vector2);
#elif defined(MLAS_NEON32_INTRINSICS)
    Vector1 = vsetq_lane_f32(vgetq_lane_f32(Vector1, 0) / vgetq_lane_f32(Vector2, 0), Vector1, 0);
    Vector1 = vsetq_lane_f32(vgetq_lane_f32(Vector1, 1) / vgetq_lane_f32(Vector2, 1), Vector1, 1);
    Vector1 = vsetq_lane_f32(vgetq_lane_f32(Vector1, 2) / vgetq_lane_f32(Vector2, 2), Vector1, 2);
    Vector1 = vsetq_lane_f32(vgetq_lane_f32(Vector1, 3) / vgetq_lane_f32(Vector2, 3), Vector1, 3);
    return Vector1;
#elif defined(MLAS_SSE2_INTRINSICS)
    return _mm_div_ps(Vector1, Vector2);
#elif defined(MLAS_WASM_SIMD_INTRINSICS)
    return wasm_f32x4_div(Vector1, Vector2);
#elif defined(MLAS_LSX_INTRINSICS)
    return __lsx_vfdiv_s(Vector1, Vector2);
#else
    return Vector1 / Vector2;
#endif
}

inline
MLAS_FLOAT32X4
MlasGreaterThanFloat32x4(MLAS_FLOAT32X4 Vector1, MLAS_FLOAT32X4 Vector2)
{
#if defined(MLAS_NEON_INTRINSICS)
    return vreinterpretq_f32_u32(vcgtq_f32(Vector1, Vector2));
#elif defined(MLAS_SSE2_INTRINSICS)
    return _mm_cmpgt_ps(Vector1, Vector2);
#elif defined(MLAS_WASM_SIMD_INTRINSICS)
    return wasm_f32x4_gt(Vector1, Vector2);
#elif defined(MLAS_VSX_INTRINSICS)
    return MLAS_FLOAT32X4(vec_cmpgt(Vector1, Vector2));
#elif defined(MLAS_LSX_INTRINSICS)
    return (MLAS_FLOAT32X4)__lsx_vfcmp_clt_s(Vector2, Vector1);
#else
    return Vector1 > Vector2;
#endif
}

inline
MLAS_FLOAT32X4
MlasAndFloat32x4(MLAS_FLOAT32X4 Vector1, MLAS_FLOAT32X4 Vector2)
{
#if defined(MLAS_SSE2_INTRINSICS)
    return _mm_and_ps(Vector1, Vector2);
#elif defined(MLAS_WASM_SIMD_INTRINSICS)
    return wasm_v128_and(Vector1, Vector2);
#elif defined(MLAS_LSX_INTRINSICS)
    return MlasReinterpretAsFloat32x4(MlasAndInt32x4(MlasReinterpretAsInt32x4(Vector1), MlasReinterpretAsInt32x4(Vector2)));
#else
    return MlasReinterpretAsFloat32x4(MlasAndInt32x4(MlasReinterpretAsInt32x4(Vector1), MlasReinterpretAsInt32x4(Vector2)));
#endif
}

inline
MLAS_FLOAT32X4
MlasOrFloat32x4(MLAS_FLOAT32X4 Vector1, MLAS_FLOAT32X4 Vector2)
{
#if defined(MLAS_SSE2_INTRINSICS)
    return _mm_or_ps(Vector1, Vector2);
#elif defined(MLAS_WASM_SIMD_INTRINSICS)
    return wasm_v128_or(Vector1, Vector2);
#elif defined(MLAS_LSX_INTRINSICS)
    return MlasReinterpretAsFloat32x4(MlasOrInt32x4(MlasReinterpretAsInt32x4(Vector1), MlasReinterpretAsInt32x4(Vector2)));
#else
    return MlasReinterpretAsFloat32x4(MlasOrInt32x4(MlasReinterpretAsInt32x4(Vector1), MlasReinterpretAsInt32x4(Vector2)));
#endif
}

inline
MLAS_FLOAT32X4
MlasAndNotFloat32x4(MLAS_FLOAT32X4 VectorNot, MLAS_FLOAT32X4 Vector)
{
#if defined(MLAS_SSE2_INTRINSICS)
    return _mm_andnot_ps(VectorNot, Vector);
#elif defined(MLAS_WASM_SIMD_INTRINSICS)
    return wasm_v128_andnot(Vector, VectorNot);
#elif defined(MLAS_LSX_INTRINSICS)
    return MlasReinterpretAsFloat32x4(MlasAndNotInt32x4(MlasReinterpretAsInt32x4(VectorNot), MlasReinterpretAsInt32x4(Vector)));
#else
    return MlasReinterpretAsFloat32x4(MlasAndNotInt32x4(MlasReinterpretAsInt32x4(VectorNot), MlasReinterpretAsInt32x4(Vector)));
#endif
}

inline
MLAS_FLOAT32X4
MlasXorFloat32x4(MLAS_FLOAT32X4 Vector1, MLAS_FLOAT32X4 Vector2)
{
#if defined(MLAS_SSE2_INTRINSICS)
    return _mm_xor_ps(Vector1, Vector2);
#elif defined(MLAS_WASM_SIMD_INTRINSICS)
    return wasm_v128_xor(Vector1, Vector2);
#elif defined(MLAS_LSX_INTRINSICS)
    return MlasReinterpretAsFloat32x4(MlasXorInt32x4(MlasReinterpretAsInt32x4(Vector1), MlasReinterpretAsInt32x4(Vector2)));
#else
    return MlasReinterpretAsFloat32x4(MlasXorInt32x4(MlasReinterpretAsInt32x4(Vector1), MlasReinterpretAsInt32x4(Vector2)));
#endif
}

inline
MLAS_FLOAT32X4
MlasBlendFloat32x4(MLAS_FLOAT32X4 Vector1, MLAS_FLOAT32X4 Vector2, MLAS_FLOAT32X4 Selection)
{
    return MlasOrFloat32x4(MlasAndFloat32x4(Vector2, Selection), MlasAndNotFloat32x4(Selection, Vector1));
}

inline
MLAS_FLOAT32X4
MlasMaximumFloat32x4(MLAS_FLOAT32X4 Vector1, MLAS_FLOAT32X4 Vector2)
{
#if defined(MLAS_NEON_INTRINSICS)
    return vmaxq_f32(Vector1, Vector2);
#elif defined(MLAS_SSE2_INTRINSICS)
    return _mm_max_ps(Vector1, Vector2);
#elif defined(MLAS_VSX_INTRINSICS)
    // Don't use vec_max to avoid undefined behavior if NAN
    return vec_sel(Vector2, Vector1, vec_cmpgt(Vector1, Vector2));
#elif defined(MLAS_WASM_SIMD_INTRINSICS)
    return wasm_f32x4_max(Vector1, Vector2);
#elif defined(MLAS_LSX_INTRINSICS)
    return __lsx_vfmax_s(Vector1, Vector2);
#else
    return MlasBlendFloat32x4(Vector2, Vector1, Vector1 > Vector2);
#endif
}

inline
MLAS_FLOAT32X4
MlasMinimumFloat32x4(MLAS_FLOAT32X4 Vector1, MLAS_FLOAT32X4 Vector2)
{
#if defined(MLAS_NEON_INTRINSICS)
    return vminq_f32(Vector1, Vector2);
#elif defined(MLAS_SSE2_INTRINSICS)
    return _mm_min_ps(Vector1, Vector2);
#elif defined(MLAS_VSX_INTRINSICS)
    // Don't use vec_min to avoid undefined behavior if NAN
    return vec_sel(Vector2, Vector1, vec_cmpgt(Vector2, Vector1));
#elif defined(MLAS_WASM_SIMD_INTRINSICS)
    return wasm_f32x4_min(Vector1, Vector2);
#elif defined(MLAS_LSX_INTRINSICS)
    return __lsx_vfmin_s(Vector1, Vector2);
#else
    return MlasBlendFloat32x4(Vector2, Vector1, Vector2 > Vector1);
#endif
}

inline
MLAS_FLOAT32X4
MlasClampFloat32x4(MLAS_FLOAT32X4 Value, float LowerRange, float UpperRange)
{
#if defined(MLAS_SSE2_INTRINSICS)
    // N.B. MINPS and MAXPS propagates the value from the second vector if the
    // value is a NaN.
#endif
    Value = MlasMaximumFloat32x4(MlasBroadcastFloat32x4(LowerRange), Value);
    Value = MlasMinimumFloat32x4(MlasBroadcastFloat32x4(UpperRange), Value);
    return Value;
}

inline
float
MlasReduceAddFloat32x4(MLAS_FLOAT32X4 Vector)
{
#if defined(MLAS_NEON64_INTRINSICS)
    Vector = vpaddq_f32(Vector, Vector);
    Vector = vpaddq_f32(Vector, Vector);
    return vgetq_lane_f32(Vector, 0);
#elif defined(MLAS_NEON32_INTRINSICS)
    float32x2_t VectorLow = vget_low_f32(Vector);
    float32x2_t VectorHigh = vget_high_f32(Vector);
    VectorLow = vpadd_f32(VectorLow, VectorHigh);
    VectorLow = vpadd_f32(VectorLow, VectorHigh);
    return vget_lane_f32(VectorLow, 0);
#elif defined(MLAS_VSX_INTRINSICS)
    Vector = MlasAddFloat32x4(Vector, MLAS_FLOAT32X4(vec_splat((__vector long long)Vector, 1)));
    Vector = MlasAddFloat32x4(Vector, vec_splat(Vector, 1));
    return Vector[0];
#else
    Vector = MlasAddFloat32x4(Vector, MlasShuffleFloat32x4<2, 3, 2, 3>(Vector));
    Vector = MlasAddFloat32x4(Vector, MlasShuffleFloat32x4<1, 1, 1, 1>(Vector));
    return MlasExtractLaneFloat32x4<0>(Vector);
#endif
}

inline
float
MlasReduceMaximumFloat32x4(MLAS_FLOAT32X4 Vector)
{
#if defined(MLAS_NEON64_INTRINSICS)
    return vmaxvq_f32(Vector);
#elif defined(MLAS_NEON32_INTRINSICS)
    float32x2_t VectorLow = vget_low_f32(Vector);
    float32x2_t VectorHigh = vget_high_f32(Vector);
    VectorLow = vpmax_f32(VectorLow, VectorHigh);
    VectorLow = vpmax_f32(VectorLow, VectorHigh);
    return vget_lane_f32(VectorLow, 0);
#elif defined(MLAS_VSX_INTRINSICS)
    Vector = MlasMaximumFloat32x4(Vector, MLAS_FLOAT32X4(vec_splat((__vector long long)Vector, 1)));
    Vector = MlasMaximumFloat32x4(Vector, vec_splat(Vector, 1));
    return Vector[0];
#else
    Vector = MlasMaximumFloat32x4(Vector, MlasShuffleFloat32x4<2, 3, 2, 3>(Vector));
    Vector = MlasMaximumFloat32x4(Vector, MlasShuffleFloat32x4<1, 1, 1, 1>(Vector));
    return MlasExtractLaneFloat32x4<0>(Vector);
#endif
}

inline
float
MlasReduceMinimumFloat32x4(MLAS_FLOAT32X4 Vector)
{
#if defined(MLAS_NEON64_INTRINSICS)
    return vminvq_f32(Vector);
#elif defined(MLAS_NEON32_INTRINSICS)
    float32x2_t VectorLow = vget_low_f32(Vector);
    float32x2_t VectorHigh = vget_high_f32(Vector);
    VectorLow = vpmin_f32(VectorLow, VectorHigh);
    VectorLow = vpmin_f32(VectorLow, VectorHigh);
    return vget_lane_f32(VectorLow, 0);
#elif defined(MLAS_VSX_INTRINSICS)
    Vector = MlasMinimumFloat32x4(Vector, MLAS_FLOAT32X4(vec_splat((__vector long long)Vector, 1)));
    Vector = MlasMinimumFloat32x4(Vector, vec_splat(Vector, 1));
    return Vector[0];
#else
    Vector = MlasMinimumFloat32x4(Vector, MlasShuffleFloat32x4<2, 3, 2, 3>(Vector));
    Vector = MlasMinimumFloat32x4(Vector, MlasShuffleFloat32x4<1, 1, 1, 1>(Vector));
    return MlasExtractLaneFloat32x4<0>(Vector);
#endif
}

// calc 2^int(N)
inline
MLAS_FLOAT32X4
MlasPowerOf2Float32x4(MLAS_FLOAT32X4 Vector)
{
    MLAS_INT32X4 emm0 = MlasAddInt32x4(MlasCastToInt32x4(Vector), MlasBroadcastInt32x4(127));
    return MlasReinterpretAsFloat32x4(MlasShiftLeftInt32x4<23>(emm0));
}

//
// Cross-platform wrappers for 64-bit vector intrinsics.
//

#if defined(MLAS_SSE2_INTRINSICS)
typedef __m128d MLAS_FLOAT64X2;
#elif defined(MLAS_VSX_INTRINSICS)
typedef __vector double MLAS_FLOAT64X2;
#elif defined(MLAS_LSX_INTRINSICS)
typedef __m128d MLAS_FLOAT64X2;
#else
#define MLAS_FLOAT64X2_UNSUPPORTED
#endif

#ifndef MLAS_FLOAT64X2_UNSUPPORTED

#if defined(MLAS_VSX_INTRINSICS)
template<unsigned Lane>
inline
double
MlasExtractLaneFloat64x2(MLAS_FLOAT64X2 Vector)
{
    return Vector[Lane];
}
inline
MLAS_FLOAT64X2
MlasMultiplyAddFloat64x2(MLAS_FLOAT64X2 Vector1, MLAS_FLOAT64X2 Vector2, MLAS_FLOAT64X2 Vector3)
{
    return vec_madd(Vector1, Vector2, Vector3);
}

inline
MLAS_FLOAT64X2
MlasBroadcastFloat64x2(const double *Value)
{
    return MLAS_FLOAT64X2{*Value, *Value};
}
#elif defined(MLAS_LSX_INTRINSICS)
template<unsigned Lane>
inline
double
MlasExtractLaneFloat64x2(MLAS_FLOAT64X2 Vector)
{
    return Vector[Lane];
}
inline
MLAS_FLOAT64X2
MlasMultiplyAddFloat64x2(MLAS_FLOAT64X2 Vector1, MLAS_FLOAT64X2 Vector2, MLAS_FLOAT64X2 Vector3)
{
    return __lsx_vfmadd_d(Vector1, Vector2, Vector3);
}

inline
MLAS_FLOAT64X2
MlasBroadcastFloat64x2(const double *Value)
{
    return MLAS_FLOAT64X2{*Value, *Value};
}
#endif
inline
MLAS_FLOAT64X2
MlasBroadcastFloat64x2(double Value)
{
#if defined(MLAS_SSE2_INTRINSICS)
    return _mm_set1_pd(Value);
#elif defined(MLAS_VSX_INTRINSICS)
    return MLAS_FLOAT64X2{Value, Value};
#elif defined(MLAS_LSX_INTRINSICS)
    return MLAS_FLOAT64X2{Value, Value};
#endif
}

inline
MLAS_FLOAT64X2
MlasZeroFloat64x2(void)
{
#if defined(MLAS_SSE2_INTRINSICS)
    return _mm_setzero_pd();
#elif defined(MLAS_VSX_INTRINSICS)
    return MlasBroadcastFloat64x2(0.0f);
#elif defined(MLAS_LSX_INTRINSICS)
    return MlasBroadcastFloat64x2(0.0f);
#endif
}

inline
MLAS_FLOAT64X2
MlasLoadFloat64x2(const double* Buffer)
{
#if defined(MLAS_SSE2_INTRINSICS)
    return _mm_loadu_pd(Buffer);
#elif defined(MLAS_VSX_INTRINSICS)
    return vec_vsx_ld(0, Buffer);
#elif defined(MLAS_LSX_INTRINSICS)
    return MLAS_FLOAT64X2(__lsx_vld((const MLAS_INT32X4 *)Buffer, 0));
#endif
}

inline
void
MlasStoreFloat64x2(double* Buffer, MLAS_FLOAT64X2 Vector)
{
#if defined(MLAS_SSE2_INTRINSICS)
    _mm_storeu_pd(Buffer, Vector);
#elif defined(MLAS_VSX_INTRINSICS)
    vec_vsx_st(Vector, 0, Buffer);
#elif defined(MLAS_LSX_INTRINSICS)
    (__lsx_vst(MLAS_INT32X4(Vector), Buffer, 0));
#endif
}

inline
void
MlasStoreAlignedFloat64x2(double* Buffer, MLAS_FLOAT64X2 Vector)
{
#if defined(MLAS_SSE2_INTRINSICS)
    _mm_store_pd(Buffer, Vector);
#elif defined(MLAS_VSX_INTRINSICS)
    *((MLAS_FLOAT64X2*)Buffer) = Vector;
#elif defined(MLAS_LSX_INTRINSICS)
    (__lsx_vst(MLAS_INT32X4(Vector), Buffer, 0));
#endif
}

inline
MLAS_FLOAT64X2
MlasMultiplyFloat64x2(MLAS_FLOAT64X2 Vector1, MLAS_FLOAT64X2 Vector2)
{
#if defined(MLAS_SSE2_INTRINSICS)
    return _mm_mul_pd(Vector1, Vector2);
#elif defined(MLAS_VSX_INTRINSICS)
    return Vector1 * Vector2;
#elif defined(MLAS_LSX_INTRINSICS)
    return __lsx_vfmul_d(Vector1, Vector2);
#endif
}

#endif  // !MLAS_FLOAT64X2_UNSUPPORTED

//
// Reads a platform specific time stamp counter.
//

inline
uint64_t
MlasReadTimeStampCounter(void)
{
#ifdef _WIN32
#if defined(MLAS_TARGET_AMD64_IX86)
    return ReadTimeStampCounter();
#else
    LARGE_INTEGER PerformanceCounter;

    QueryPerformanceCounter(&PerformanceCounter);

    return (ULONG64)PerformanceCounter.QuadPart;
#endif
#else
#if defined(MLAS_TARGET_AMD64)
    uint32_t eax, edx;

    __asm__ __volatile__
    (
        "rdtsc"
        : "=a" (eax), "=d" (edx)
    );

    return ((uint64_t)edx << 32) | eax;
#elif defined(MLAS_TARGET_LARCH64)
    uint64_t time_cnt, id;

    __asm__ __volatile__
    (
        "rdtime.d %0, %1\n\t"
        : "=r" (time_cnt), "=r" (id)
	::
    );

    return time_cnt;
#else
    return 0;
#endif
#endif
}

//
// Aligned buffer for GEMM packing, etc.
//


constexpr size_t ThreadedBufAlignment = 64;
extern thread_local size_t ThreadedBufSize;
#ifdef _MSC_VER
extern thread_local std::unique_ptr<uint8_t, decltype(&_aligned_free)> ThreadedBufHolder;
#else
extern thread_local std::unique_ptr<uint8_t, decltype(&free)> ThreadedBufHolder;
#endif

inline
constexpr size_t
UpAlignSize(size_t size)
{
    size = (size + ThreadedBufAlignment - 1) / ThreadedBufAlignment;
    return size * ThreadedBufAlignment;
}


inline
void
MlasThreadedBufAlloc(size_t size)
{
    if (size > ThreadedBufSize) {
#ifdef _MSC_VER
        ThreadedBufHolder.reset(
            reinterpret_cast<uint8_t*>(_aligned_malloc(size, ThreadedBufAlignment)));
#elif (__STDC_VERSION__ >= 201112L) && !defined(__APPLE__)
        ThreadedBufHolder.reset(
            reinterpret_cast<uint8_t*>(aligned_alloc(ThreadedBufAlignment, size)));
#else
	// aligned_alloc unavailable macos 10.14 or earlier
        void* ptr;
        int err = posix_memalign(&ptr, ThreadedBufAlignment, size);
        if (err != 0) {
            ptr = nullptr;
        }
        ThreadedBufHolder.reset(reinterpret_cast<uint8_t*>(ptr));
#endif

        ThreadedBufSize = size;
    }
}

//
// Utilities for INT4 quantization.
//

template<bool Signed>
struct Int4Traits;

template<>
struct Int4Traits<true> {
    using UnpackedType = int8_t;
    static constexpr int8_t Min = -8;
    static constexpr int8_t Max = 7;
};

template<>
struct Int4Traits<false> {
    using UnpackedType = uint8_t;
    static constexpr int8_t Min = 0;
    static constexpr int8_t Max = 15;
};

template<typename UnpackedType>
inline
void
MlasSetInt4Element(uint8_t* Output, size_t ElemIndex, UnpackedType Value)
{
    static_assert(std::is_same_v<UnpackedType, uint8_t> || std::is_same_v<UnpackedType, int8_t>);

    const size_t OutputIndex = ElemIndex >> 1;  // which byte
    const size_t NibbleIndex = ElemIndex & 0x1; // which 4-bit elem in the byte
    const uint8_t Shift = static_cast<uint8_t>(NibbleIndex << 2); // Either 0 or 4
    const uint8_t Mask = static_cast<uint8_t>(0xF0 >> Shift);
    uint8_t* Dst = &Output[OutputIndex];

    *Dst &= Mask; // Clear 4-bit lane
    *Dst |= static_cast<uint8_t>((Value & 0xF) << Shift); // Set 4-bit lane
}

template<typename UnpackedType>
inline
void
MlasPackInt4Elements(uint8_t* Output, UnpackedType ValueLow, UnpackedType ValueHigh)
{
    static_assert(std::is_same_v<UnpackedType, uint8_t> || std::is_same_v<UnpackedType, int8_t>);
    *Output = static_cast<uint8_t>(((ValueHigh & 0xF) << 4) | (ValueLow & 0xF));
}
