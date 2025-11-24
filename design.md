# Minimal CUDA MLP Implementation

This repository hosts a barebones CUDA/C++ implementation of a multilayer perceptron (MLP) composed of fully connected layers with bias and GELU activation.

## Goals

- Define networks purely by a vector of layer sizes.
- Provide CUDA kernels for forward and backward propagation (linear + GELU).
- Support serialization/deserialization of model parameters and metadata (layer sizes) in a portable binary format.
- Keep the implementation dependency-light for ease of reading and modification.

## Notes for future work

- Training monitors: we expect to add logging compatible with Weights & Biases (wandb) by emitting per-step metrics to a simple log file that a lightweight Python script can stream to wandb. This keeps the C++/CUDA core free of Python/runtime dependencies while enabling experiment tracking.
- Datasets: once the core is stable, add dataset loaders and simple training loops for MNIST, CIFAR-10, etc., using the same linear+GELU stack.
- Optimizers: the initial version only accumulates gradients; simple SGD (and later Adam) updates can be layered on top.
