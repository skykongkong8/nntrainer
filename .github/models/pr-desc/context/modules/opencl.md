# opencl

## Responsibility
OpenCL device/context, kernels, and program management for GPU offload.

## Key Files
- `opencl_context_manager.*`, `opencl_command_queue_manager.*`, `opencl_device_info.*`
- `opencl_program.*`, `opencl_kernel.*`, `opencl_buffer.*`
- Vendor headers under `opencl/CL/*`

## Integration
- Layers route eligible ops through OpenCL paths
- Tensor provides OpenCL buffers where needed
