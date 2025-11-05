# tensor_cpu_backend

## Responsibility
High-performance CPU kernels (SIMD/ISA-specific) for tensor ops and quantized math.

## Layout
- `cpu_backend.h`, `README.md`, `meson.build`
- **CBLAS bridge**: `cblas_interface/*` for BLAS fallback/interop
- **Fallback**: generic portable paths `fallback/*` (+ fp16, kleidiai)
- **x86/AVX2**: `x86/avx2_impl.cpp`, `x86/vnni_impl.cpp`, transpose/pack kernels
- **ARM/NEON**: `arm/*` including `dotprod`, `hgemm`, `matrix_transpose_neon/*`, `kai/*`
- **GGML interface**: `ggml_interface/nntr_ggml_impl/*` for shared kernels and quant ops

## Notable Ops
- GEMM (fp32/fp16/int4/int8 paths), pack/unpack, activation, layernorm, softmax
- Quant utilities: q4_K, q6_K conversions and dot-products

## Dispatch
- Selected at build-time or runtime by ISA probing; clean fallback when SIMD absent
