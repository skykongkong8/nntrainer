# schema

## Responsibility
Model/Op schema definitions for configuration and serialization.

## Files
- `nntrainer_schema.fbs`, `tf_schema.fbs`: FlatBuffers schemas
- `onnx.proto`: ONNX proto for interchange
- `meson.build` for codegen targets

## Usage
- Interpreters in `compiler` validate/translate configs against these schemas
