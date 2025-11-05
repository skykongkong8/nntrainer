# utils

## Responsibility
Cross-cutting services: properties, threading, profiling, platform adapters.

## Highlights
- Properties: `base_properties.*`
- Thread pool: `bs_thread_pool.h`, `bs_thread_pool_manager.*`, `nntr_threads.*`
- Profiler: `profiler.*`
- Platform shims: `mman_windows.*`, `dynamic_library_loader.h`
- FP16 conversion: `fp16.*`
- Config helpers: `ini_wrapper.*`

## Role
Enable portability and performance without entangling core modules.
