# NNTrainer Architecture Overview

## Purpose
On-device training framework with modular graph/layer/tensor pipeline optimized for low-memory devices.

## Lifecycle
1) **Load/Configure** → models, graph, layers
2) **Compile** → compiler (realizers, interpreters, graph optimizers)
3) **Initialize** → tensor (tensor pool, memory planner), utils (threads, profiler)
4) **Train/Evaluate** → layers + tensor backends (CPU/OpenCL), optimizers, dataset

```mermaid
flowchart LR
    A[User API / Model Config] --> B(models:model_loader.cpp)
    B --> C(graph:network_graph.cpp)
    C --> D(compiler:*_realizer, onnx/ini interpreter)
    D --> E(tensor:memory planner & pools)
    E --> F{Train Loop}
    subgraph Exec
    F --> L[layers: forward/backward]
    L --> T[tensor ops]
    T -->|CPU| CB[tensor/cpu_backend/*]
    T -->|OpenCL| CL[opencl/*]
    F --> O[optimizers/*]
    F --> DS[dataset/*]
    end
```
