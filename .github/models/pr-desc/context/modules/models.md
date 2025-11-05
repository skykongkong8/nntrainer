# models

## Responsibility
Top-level lifecycle orchestration and model representation.

## Key Files
- `model_loader.*`: parse configs to layer specs
- `neuralnet.*`: model facade coordinating phases
- `model_common_properties.*`: hyperparameters and defaults
- `dynamic_training_optimization.*`: runtime heuristics

## Flow
`Model::load()` → `Graph::build()` → `Compiler::compile()` → `Initialize(TensorPool)` → `Train(optim + dataset)`
