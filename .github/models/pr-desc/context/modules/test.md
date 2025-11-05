# test

## Scope
Unit and integration tests spanning API, tensor backends, planners, layers, models, and C-API.

## Sample Entries (from provided test archive)
- `unittest_nntrainer_tensor.cpp`, `unittest_nntrainer_tensor_fp16.cpp`
- `unittest_nntrainer_cpu_backend{_fp16}.cpp`
- `unittest_nntrainer_graph.cpp`, `unittest_nntrainer_models.cpp`, `unittest_nntrainer_exe_order.cpp`
- `tizen_capi/unittest_tizen_capi*.cpp`
- Example configs: `test_models/models/*.ini`, `*.tflite`

## Execution
Meson/CMake targets integrate with CI; coverage scripts: `unittestcoverage.py`
