# optimizers

## Responsibility
Parameter update rules and LR schedulers.

## Key Files
- `adam.*`, `adamw.*`, optimizer base: `optimizer_devel.*`, `optimizer_wrapped.cpp`, `optimizer_context.*`
- LR schedulers: `lr_scheduler_{constant,cosine,exponential,linear,step}.*`

## Protocol
- `apply(grad, weight, state)` per tensor
- LR schedule resolved via `optimizer_context`
