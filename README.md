# muon-diy-implementation

A barebones C++/CUDA multilayer perceptron consisting of fully connected linear layers with bias and GELU activation. Networks are parameterized by a list of layer sizes and expose forward/backward, SGD updates, and binary serialization.

## Building

```bash
mkdir -p build && cd build
cmake ..
cmake --build .
```

This produces a static library `libmuon_nn.a` exposing the neural network interface in `include/nn.h`. CUDA is required for compilation.

### XOR example

To build the provided XOR training demo:

```bash
mkdir -p build && cd build
cmake ..
cmake --build . --target xor_example
./xor_example
```

### MNIST example

1. Prepare the dataset (writes normalized binaries under `datasets/mnist/processed`).
   The script can consume either the original IDX gzip archives **or** the
   HuggingFace parquet export (`train-00000-of-00001.parquet` and
   `test-00000-of-00001.parquet`). If you already have the dataset, place it
   under `datasets/mnist/source/` (with or without git metadata). Otherwise, it
   will attempt to clone `https://huggingface.co/datasets/ylecun/mnist` using
   git-lfs.

   ```bash
   # Optional: only required if cloning the HuggingFace repository
   git lfs install --system

   # Required when reading parquet exports
   pip install pyarrow pillow
   python3 tools/prepare_mnist.py
   ```

2. Build and run the MNIST demo (requires CUDA):

   ```bash
   mkdir -p build && cd build
   cmake ..
   cmake --build . --target mnist_example
   ./mnist_example
   ```

## Usage sketch

```cpp
#include "nn.h"

int main() {
    // Define a 3-layer MLP: input 784 -> hidden 128 -> output 10.
    muon::NeuralNetwork net({784, 128, 10});
    muon::ForwardContext ctx;

    // Allocate device input/output/gradient buffers before calling forward/backward.
    // float* device_input = ...;
    // float* grad_output = ...; // upstream gradient at network output

    // auto output = net.forward(device_input, batch_size, ctx);
    // net.backward(device_input, ctx, grad_output, grad_input, batch_size);
    // net.sgd_update(learning_rate);
    // or
    // net.adamw_update(learning_rate);

    // Persist parameters to disk.
    // net.save("mlp.bin");
    // net.load("mlp.bin");
}
```

See `design.md` for notes on future logging and dataset integrations.
