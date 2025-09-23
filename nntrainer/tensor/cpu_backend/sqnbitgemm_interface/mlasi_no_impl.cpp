#include "mlasi.h"
#include "mlas_q4.h"
#include <cstddef>
#include <cmath>
#include <cstdint>
#include <cstring>

// 1. Float GEMM operation (basic implementation)
void MlasSgemmOperation(
    CBLAS_TRANSPOSE TransA,
    CBLAS_TRANSPOSE TransB,
    size_t M,
    size_t N,
    size_t K,
    float alpha,
    const float* A,
    size_t lda,
    const float* B,
    size_t ldb,
    float beta,
    float* C,
    size_t ldc)
{
    // Iterate over output matrix C of size MxN
    for (size_t i = 0; i < M; ++i) {
        for (size_t j = 0; j < N; ++j) {
            // Scale existing C value by beta
            float accum = beta * C[i * ldc + j];
            // Compute dot product of row i of A and col j of B
            for (size_t k = 0; k < K; ++k) {
                float a_val = TransA == CblasNoTrans ? A[i * lda + k] : A[k * lda + i];
                float b_val = TransB == CblasNoTrans ? B[k * ldb + j] : B[j * ldb + k];
                accum += alpha * (a_val * b_val);
            }
            C[i * ldc + j] = accum;
        }
    }
}

// 2. Depthwise convolution (uint8_t x uint8_t)
template<typename InputType, typename FilterType>
void MlasConvDepthwiseKernel(
    const InputType* const* Input, InputType InputZeroPoint,
    const FilterType* Filter, FilterType FilterZeroPoint,
    int32_t* Output,
    size_t Channels, size_t OutputCount, size_t KernelSize)
{
    // Naive depthwise convolution: for each channel, sum element-wise products
    // Assume Input is an array of pointers to each input patch per output element
    for (size_t c = 0; c < Channels; ++c) {
        for (size_t out_idx = 0; out_idx < OutputCount; ++out_idx) {
            // Compute one output element (per channel)
            int32_t sum = 0;
            const InputType* in_ptr = Input[out_idx] + c * KernelSize;
            const FilterType* filt_ptr = Filter + c * KernelSize;
            for (size_t k = 0; k < KernelSize; ++k) {
                // subtract zero-points and accumulate
                int32_t in_val = static_cast<int32_t>(in_ptr[k]) - static_cast<int32_t>(InputZeroPoint);
                int32_t fil_val = static_cast<int32_t>(filt_ptr[k]) - static_cast<int32_t>(FilterZeroPoint);
                sum += in_val * fil_val;
            }
            Output[c + out_idx * Channels] = sum;
        }
    }
}
// Explicit template instantiations for needed type combinations
template void MlasConvDepthwiseKernel<uint8_t, uint8_t>(
    const uint8_t* const* Input, uint8_t InputZeroPoint,
    const uint8_t* Filter, uint8_t FilterZeroPoint,
    int32_t* Output, size_t Channels, size_t OutputCount, size_t KernelSize);
template void MlasConvDepthwiseKernel<uint8_t, int8_t>(
    const uint8_t* const* Input, uint8_t InputZeroPoint,
    const int8_t* Filter, int8_t FilterZeroPoint,
    int32_t* Output, size_t Channels, size_t OutputCount, size_t KernelSize);

// 3. Depthwise convolution AVX2 variant (calls baseline implementation)
template<typename InputType, typename FilterType>
void MlasConvDepthwiseKernelAvx2(
    const InputType* const* Input, InputType InputZeroPoint,
    const FilterType* Filter, FilterType FilterZeroPoint,
    int32_t* Output,
    size_t Channels, size_t OutputCount, size_t KernelSize)
{
    // For now, reuse the baseline implementation
    MlasConvDepthwiseKernel<InputType, FilterType>(
        Input, InputZeroPoint, Filter, FilterZeroPoint,
        Output, Channels, OutputCount, KernelSize);
}
template void MLASCALL MlasConvDepthwiseKernelAvx2<uint8_t, uint8_t>(
    const uint8_t* const* Input, uint8_t InputZeroPoint,
    const uint8_t* Filter, uint8_t FilterZeroPoint,
    int32_t* Output, size_t Channels, size_t OutputCount, size_t KernelSize);
template void MLASCALL MlasConvDepthwiseKernelAvx2<uint8_t, int8_t>(
    const uint8_t* const* Input, uint8_t InputZeroPoint,
    const int8_t* Filter, int8_t FilterZeroPoint,
    int32_t* Output, size_t Channels, size_t OutputCount, size_t KernelSize);

// 4. Depthwise convolution for float (CHW layout)
void MlasConvDepthwiseFloat_CHW(
    const MLAS_CONV_PARAMETERS* Parameters,
    const float* Input,
    const float* Filter,
    float* Output,
    const float* Zeros)
{
    // Basic depthwise conv: assume 2D conv for simplicity
    size_t C = Parameters->InputChannels; // number of channels (depthwise: equals output channels)
    size_t inH = Parameters->InputShape[0];
    size_t inW = Parameters->InputShape[1];
    size_t outH = Parameters->OutputShape[0];
    size_t outW = Parameters->OutputShape[1];
    size_t strideH = Parameters->StrideShape[0];
    size_t strideW = Parameters->StrideShape[1];
    size_t padH = Parameters->Padding[0]; // top padding
    size_t padW = Parameters->Padding[2]; // left padding
    size_t kernelH = Parameters->KernelShape[0];
    size_t kernelW = Parameters->KernelShape[1];
    // Iterate over batch, channels, and output spatial positions
    for (size_t n = 0; n < Parameters->BatchCount; ++n) {
        const float* batchInput = Input + n * Parameters->InputSize;
        float* batchOutput = Output + n * Parameters->OutputSize;
        for (size_t c = 0; c < C; ++c) {
            const float* inChannel = batchInput + c * (inH * inW);
            const float* filtChannel = Filter + c * (kernelH * kernelW);
            float* outChannel = batchOutput + c * (outH * outW);
            for (size_t oh = 0; oh < outH; ++oh) {
                for (size_t ow = 0; ow < outW; ++ow) {
                    // Apply Beta factor to existing output (accumulate if Beta != 0)
                    size_t outIndex = oh * outW + ow;
                    float value = Parameters->Beta * outChannel[outIndex];
                    // Convolution sum for one output position
                    for (size_t kh = 0; kh < kernelH; ++kh) {
                        size_t ih = oh * strideH + kh - padH;
                        if (ih >= inH) continue;
                        for (size_t kw = 0; kw < kernelW; ++kw) {
                            size_t iw = ow * strideW + kw - padW;
                            if (iw >= inW) continue;
                            float in_val = inChannel[ih * inW + iw];
                            float filt_val = filtChannel[kh * kernelW + kw];
                            value += in_val * filt_val;
                        }
                    }
                    outChannel[outIndex] = value;
                }
            }
        }
    }
}

// 5. 4-bit quantization support functions
size_t MlasQ4GemmPackBSize(MLAS_BLK_QUANT_TYPE QType, size_t N, size_t K)
{
    // Return a conservative buffer size (bytes) for packing B matrix
    // (Half byte per element + some overhead per column)
    size_t bytes_per_column = (K + 1) / 2; // K values compressed into bytes (2 per byte)
    // Assume some fixed overhead per column for scale/zero-point (e.g., 16 bytes)
    size_t overhead_per_column = 16;
    return N * (bytes_per_column + overhead_per_column);
}

void* MlasQ4GemmPackB(
    MLAS_BLK_QUANT_TYPE QType,
    void* PackedBuf,
    const float* FpData,
    size_t N,
    size_t K,
    size_t ldb)
{
    // Pack B (FpData of size KxN) into 4-bit PackedBuf.
    uint8_t* out = reinterpret_cast<uint8_t*>(PackedBuf);
    size_t byteIndex = 0;
    for (size_t j = 0; j < N; ++j) {
        for (size_t i = 0; i < K; i += 2) {
            // Quantize two values into one byte
            int8_t qval1 = static_cast<int8_t>(lroundf(std::fmax(-8.f, std::fmin(7.f, FpData[i + j*ldb]))));
            int8_t qval2 = 0;
            if (i + 1 < K) {
                qval2 = static_cast<int8_t>(lroundf(std::fmax(-8.f, std::fmin(7.f, FpData[i+1 + j*ldb]))));
            }
            // Convert to 4-bit (0~15) representation (assuming qval already -8~7)
            uint8_t nibble1 = static_cast<uint8_t>(qval1 & 0x0F);
            uint8_t nibble2 = static_cast<uint8_t>(qval2 & 0x0F);
            // Combine into one byte (qval1 as low nibble, qval2 as high nibble)
            out[byteIndex++] = (nibble2 << 4) | (nibble1);
        }
        // (Note: We ignore storing scale/zero-point here for simplicity)
    }
    return out + byteIndex;
}

void MlasQ4GemmUnPackB(
    MLAS_BLK_QUANT_TYPE QType,
    const void* PackedBuf,
    float* FpData,
    size_t N,
    size_t K,
    size_t ldb)
{
    // Unpack the 4-bit packed data back into float (approximate).
    const uint8_t* in = reinterpret_cast<const uint8_t*>(PackedBuf);
    size_t byteIndex = 0;
    for (size_t j = 0; j < N; ++j) {
        for (size_t i = 0; i < K; i += 2) {
            uint8_t byte = in[byteIndex++];
            int8_t low = static_cast<int8_t>(byte & 0x0F);
            int8_t high = static_cast<int8_t>((byte >> 4) & 0x0F);
            // Convert 4-bit back to signed int (-8~7)
            if (low >= 8) low -= 16;
            if (high >= 8) high -= 16;
            FpData[i + j*ldb] = static_cast<float>(low);
            if (i + 1 < K) {
                FpData[i+1 + j*ldb] = static_cast<float>(high);
            }
        }
    }
    // (Note: result is quantized integer values as float, not original values)
}

// 6. 4-bit GEMM computation (8-bit A, 4-bit B)
void MlasQ8Q4GemmBatch(
    size_t M,
    size_t N,
    size_t K,
    const int8_t* A,
    size_t lda,
    int8_t A_zero_point,
    const void* PackedB,
    int8_t B_zero_point,
    int32_t* C,
    size_t ldc,
    size_t batch_count)
{
    // For simplicity, ignore batch_count (assume 1) and zero points.
    const uint8_t* Bdata = reinterpret_cast<const uint8_t*>(PackedB);
    for (size_t i = 0; i < M; ++i) {
        for (size_t j = 0; j < N; ++j) {
            int32_t sum = 0;
            // Compute dot product of A[i,:] and B[:,j]
            for (size_t k = 0; k < K; ++k) {
                // A element (already int8)
                int32_t a_val = static_cast<int32_t>(A[i * lda + k]);
                // B element (packed 4-bit)
                size_t byteIndex = (j * K + k) / 2;
                bool lowNibble = ((j * K + k) % 2 == 0);
                uint8_t byte = Bdata[byteIndex];
                int8_t b_val4 = lowNibble ? (byte & 0x0F) : ((byte >> 4) & 0x0F);
                if (b_val4 >= 8) b_val4 -= 16; // sign extend 4-bit
                sum += a_val * b_val4;
            }
            C[i * ldc + j] = sum;
        }
    }
}

void MlasQ4GemmBatch( // (if needed, just call Q8Q4 implementation assuming A is also 8-bit)
    size_t M,
    size_t N,
    size_t K,
    const int8_t* A,
    size_t lda,
    int8_t A_zero_point,
    const void* PackedB,
    int8_t B_zero_point,
    int32_t* C,
    size_t ldc,
    size_t batch_count)
{
    // Use the same implementation as Q8Q4
    MlasQ8Q4GemmBatch(M, N, K, A, lda, A_zero_point, PackedB, B_zero_point, C, ldc, batch_count);
}
