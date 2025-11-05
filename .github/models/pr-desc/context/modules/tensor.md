# tensor

## Responsibility
Tensor abstraction, memory planning, caching, quantized tensor types, and device operations.

## Major Components
- **Tensor core types**: `float_tensor.*`, `char_tensor.*`, `short_tensor.*`
- **Quantized formats**: `q4_0_tensor.*`, `q4_k_tensor.*`, `q6_k_tensor.*`, `bcq_tensor.*`, `quantizer.*`
- **Tensor Pool & Memory Planner**: `cache_pool.*`, `cache_loader.*`, `cache_elem.*`, `basic_planner.*`, `optimized_v{1,2,3}_planner.*`
- **CL operations**: `cl_operations/*` attention/blas kernels and interfaces
- **Backend bridge**: `cpu_backend/*` (separate doc)

## Execution
- Layers request tensors in `finalize()`; planner assigns offsets in shared arenas
- Lazy tensors and cache loader reduce peak memory; execution order decides reuse
