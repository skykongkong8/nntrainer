# dataset

## Responsibility
Data ingress and mini-batch production.

## Key Files
- `databuffer.*`, `databuffer_factory.*`: unified buffering
- `data_producer.h`, `dir_data_producers.*`, `func_data_producer.*`, `random_data_producers.*`, `raw_file_data_producer.*`
- `data_iteration.*`, `iteration_queue.*`: iteration lifecycle

## Interfaces
- Pushes input/label tensors to the train loop
- Pluggable producers; factory for construction

## Constraints
- Back-pressure via queues; decoupled from compute thread-pool
