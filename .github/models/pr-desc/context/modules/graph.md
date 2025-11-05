# graph

## Responsibility
Network topology and execution ordering carriers.

## Core Files
- `network_graph.*`: high-level graph
- `graph_core.*`, `graph_node.h`
- `connection.*`: edge definitions

## Behavior
- Builds nodes from layer specs during Configure
- Supplies DAG to compiler for realization/ordering
