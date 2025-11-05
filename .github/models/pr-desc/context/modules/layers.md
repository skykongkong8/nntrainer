# layers

## Responsibility
Operators with forward/backward, shapes, and weights.

## Representative Files
- `activation_layer.*`, `addition_layer.*`, `attention_layer.*`, `bn_layer.*`, `centroid_knn.*`, `channel_shuffle.*`
- `cl_layers/*` OpenCL counterparts for selected layers
- `fc_layer_*`, `conv/pooling/*` (as applicable)

## Contracts
- `finalize()` for shape/tensor requests
- `forward() / backward()` use tensor ops, not raw pointers
- Properties via base property system (see utils/base_properties.*)

## Extensibility
- New layer class + registration; gains compile-time realization and runtime scheduling
