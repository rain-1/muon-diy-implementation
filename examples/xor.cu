#include "nn.h"

#include <cuda_runtime.h>

#include <cmath>
#include <iostream>
#include <stdexcept>
#include <vector>

using muon::ForwardContext;
using muon::NeuralNetwork;

namespace {

inline void check_cuda(cudaError_t err, const char* msg) {
    if (err != cudaSuccess) {
        throw std::runtime_error(std::string(msg) + ": " + cudaGetErrorString(err));
    }
}

// Compute gradient of mean squared error w.r.t. network outputs.
__global__ void mse_grad_kernel(const float* preds, const float* targets, float* grad, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    grad[idx] = 2.0f * (preds[idx] - targets[idx]) / static_cast<float>(n);
}

float compute_loss(const std::vector<float>& preds, const std::vector<float>& targets) {
    float acc = 0.0f;
    for (size_t i = 0; i < preds.size(); ++i) {
        float diff = preds[i] - targets[i];
        acc += diff * diff;
    }
    return acc / static_cast<float>(preds.size());
}

}  // namespace

int main() {
    // XOR dataset: four samples, input dimension 2, output dimension 1.
    std::vector<float> host_inputs = {
        0.0f, 0.0f,
        0.0f, 1.0f,
        1.0f, 0.0f,
        1.0f, 1.0f,
    };
    std::vector<float> host_targets = {0.0f, 1.0f, 1.0f, 0.0f};
    const int batch = 4;
    const int input_dim = 2;
    const int output_dim = 1;

    // Simple two-layer network: 2 -> 4 -> 1.
    NeuralNetwork net({input_dim, 4, output_dim});

    float* device_inputs = nullptr;
    float* device_targets = nullptr;
    float* device_grad_output = nullptr;
    float* device_grad_input = nullptr;

    check_cuda(cudaMalloc(&device_inputs, host_inputs.size() * sizeof(float)), "alloc inputs");
    check_cuda(cudaMalloc(&device_targets, host_targets.size() * sizeof(float)), "alloc targets");
    check_cuda(cudaMalloc(&device_grad_output, host_targets.size() * sizeof(float)), "alloc grad output");
    check_cuda(cudaMalloc(&device_grad_input, host_inputs.size() * sizeof(float)), "alloc grad input");

    check_cuda(cudaMemcpy(device_inputs, host_inputs.data(), host_inputs.size() * sizeof(float), cudaMemcpyHostToDevice),
               "copy inputs");
    check_cuda(
        cudaMemcpy(device_targets, host_targets.data(), host_targets.size() * sizeof(float), cudaMemcpyHostToDevice),
        "copy targets");

    const float learning_rate = 0.1f;
    const int epochs = 1000;

    for (int epoch = 0; epoch < epochs; ++epoch) {
        ForwardContext ctx;
        float* device_output = net.forward(device_inputs, batch, ctx);

        // Pull predictions back to host for loss computation.
        std::vector<float> host_output(host_targets.size());
        check_cuda(cudaMemcpy(host_output.data(), device_output, host_output.size() * sizeof(float), cudaMemcpyDeviceToHost),
                   "read output");

        float loss = compute_loss(host_output, host_targets);

        // Compute dL/dy on device.
        int threads = 256;
        int blocks = (batch * output_dim + threads - 1) / threads;
        mse_grad_kernel<<<blocks, threads>>>(device_output, device_targets, device_grad_output, batch * output_dim);

        net.backward(device_inputs, ctx, device_grad_output, device_grad_input, batch);
        net.sgd_update(learning_rate);

        muon::NeuralNetwork::release_context(ctx);

        if (epoch % 100 == 0) {
            std::cout << "Epoch " << epoch << " loss: " << loss << "\n";
        }
    }

    cudaFree(device_inputs);
    cudaFree(device_targets);
    cudaFree(device_grad_output);
    cudaFree(device_grad_input);
    return 0;
}

