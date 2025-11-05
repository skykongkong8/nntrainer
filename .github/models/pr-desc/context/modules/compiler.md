# compiler

## Responsibility
Graph finalization and lowering. Realizers transform/insert layers; interpreters parse model formats; compilation computes execution order before tensor allocation.

## Key Subpackages / Files
- `activation_realizer.*`, `bn_realizer.*`, `flatten_realizer.*`, `input_realizer.*`, `multiout_realizer.*`, `previous_input_realizer.*`: structure edits
- `ini_interpreter.*`, `onnx_interpreter.*`, `flatbuffer_opnode.*`, `interpreter.h`: model format parsing
- `compiler.h`, `compiler_fwd.h`: compile entrypoints and context

## Inputs/Outputs
- **In:** Graph constructed from `models`/`graph`
- **Out:** Optimized graph with concrete layer wiring; metadata for tensor planning

## Architectural Notes
- Separation of *interpretation* (syntax → IR) and *realization* (IR → executable graph)
- Compile-time only; no runtime state beyond derived graph order
