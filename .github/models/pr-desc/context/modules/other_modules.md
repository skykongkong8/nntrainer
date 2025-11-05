# other_modules

## Memory & Execution Orchestration
- Memory planners (basic/optimized v1–v3) determine buffer lifetimes and arena placement.
- Cache loader swaps tensor payloads based on execution order and reuse distance.

## Device Abstraction
- OpenCL paths coexist with CPU backends; CL ops live under `tensor/cl_operations` and `opencl/*` device layer.

## Extensibility Points
- New layer/optimizer: add class + registration; compiler realizes structure; tensor allocates; tests extend.
- Interpreters: implement in `compiler/*_interpreter.*` and bind to schemas.
