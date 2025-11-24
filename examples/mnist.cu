#include "nn.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

using muon::ForwardContext;
using muon::NeuralNetwork;

namespace {

inline void check_cuda(cudaError_t err, const char* msg) {
    if (err != cudaSuccess) {
        throw std::runtime_error(std::string(msg) + ": " + cudaGetErrorString(err));
    }
}

struct Tensor {
    std::vector<int> shape;
    std::vector<float> data;
};

Tensor load_tensor(const std::string& path) {
    std::ifstream file(path, std::ios::binary);
    if (!file) {
        throw std::runtime_error("Failed to open tensor file: " + path);
    }

    int32_t ndims = 0;
    file.read(reinterpret_cast<char*>(&ndims), sizeof(int32_t));
    if (!file || ndims <= 0) {
        throw std::runtime_error("Malformed tensor header in: " + path);
    }

    std::vector<int> shape(ndims);
    file.read(reinterpret_cast<char*>(shape.data()), ndims * sizeof(int32_t));
    if (!file) {
        throw std::runtime_error("Failed to read tensor shape from: " + path);
    }

    size_t total = 1;
    for (int dim : shape) {
        if (dim <= 0) {
            throw std::runtime_error("Invalid tensor dimension in: " + path);
        }
        total *= static_cast<size_t>(dim);
    }

    std::vector<float> data(total);
    file.read(reinterpret_cast<char*>(data.data()), total * sizeof(float));
    if (static_cast<size_t>(file.gcount()) != total * sizeof(float)) {
        throw std::runtime_error("Tensor data size mismatch in: " + path);
    }

    return {std::move(shape), std::move(data)};
}

__global__ void mse_grad_kernel(const float* preds, const float* targets, float* grad, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    grad[idx] = 2.0f * (preds[idx] - targets[idx]) / static_cast<float>(n);
}

float mse_loss(const std::vector<float>& preds, const std::vector<float>& targets) {
    if (preds.size() != targets.size()) {
        throw std::runtime_error("Loss vectors must match in size");
    }
    float acc = 0.0f;
    for (size_t i = 0; i < preds.size(); ++i) {
        float diff = preds[i] - targets[i];
        acc += diff * diff;
    }
    return acc / static_cast<float>(preds.size());
}

float batch_accuracy(const std::vector<float>& preds, const std::vector<float>& targets, int batch, int classes) {
    int correct = 0;
    for (int i = 0; i < batch; ++i) {
        int pred_idx = 0;
        float pred_val = preds[i * classes];
        for (int c = 1; c < classes; ++c) {
            float val = preds[i * classes + c];
            if (val > pred_val) {
                pred_val = val;
                pred_idx = c;
            }
        }

        int target_idx = 0;
        float target_val = targets[i * classes];
        for (int c = 1; c < classes; ++c) {
            float val = targets[i * classes + c];
            if (val > target_val) {
                target_val = val;
                target_idx = c;
            }
        }

        if (pred_idx == target_idx) {
            ++correct;
        }
    }
    return static_cast<float>(correct) / static_cast<float>(batch);
}

}  // namespace

int main() {
    const std::string dataset_root = "datasets/mnist/processed";
    const std::string train_images_path = dataset_root + "/train_images.bin";
    const std::string train_labels_path = dataset_root + "/train_labels.bin";
    const std::string test_images_path = dataset_root + "/test_images.bin";
    const std::string test_labels_path = dataset_root + "/test_labels.bin";

    Tensor train_images = load_tensor(train_images_path);
    Tensor train_labels = load_tensor(train_labels_path);
    Tensor test_images = load_tensor(test_images_path);
    Tensor test_labels = load_tensor(test_labels_path);

    if (train_images.shape.size() != 2 || train_images.shape[1] != 28 * 28) {
        throw std::runtime_error("Train images must have shape [N, 784]");
    }
    if (train_labels.shape.size() != 2 || train_labels.shape[1] != 10) {
        throw std::runtime_error("Train labels must have shape [N, 10]");
    }

    const int train_samples = train_images.shape[0];
    const int input_dim = train_images.shape[1];
    const int num_classes = train_labels.shape[1];

    NeuralNetwork net({input_dim, 256, 128, 64, num_classes});

    const int batch_size = 128;
    const int epochs = 2;
    const float learning_rate = 0.001f;
    const float beta1 = 0.9f;
    const float beta2 = 0.999f;
    const float eps = 1e-8f;
    const float weight_decay = 0.01f;
    const int steps_per_epoch = train_samples / batch_size;

    float* device_inputs = nullptr;
    float* device_targets = nullptr;
    float* device_grad_output = nullptr;
    float* device_grad_input = nullptr;

    check_cuda(cudaMalloc(&device_inputs, static_cast<size_t>(batch_size * input_dim) * sizeof(float)), "alloc inputs");
    check_cuda(cudaMalloc(&device_targets, static_cast<size_t>(batch_size * num_classes) * sizeof(float)), "alloc targets");
    check_cuda(cudaMalloc(&device_grad_output, static_cast<size_t>(batch_size * num_classes) * sizeof(float)),
               "alloc grad output");
    check_cuda(cudaMalloc(&device_grad_input, static_cast<size_t>(batch_size * input_dim) * sizeof(float)),
               "alloc grad input");

    for (int epoch = 0; epoch < epochs; ++epoch) {
        float epoch_loss = 0.0f;
        float epoch_acc = 0.0f;

        for (int step = 0; step < steps_per_epoch; ++step) {
            int offset = step * batch_size * input_dim;
            int label_offset = step * batch_size * num_classes;

            check_cuda(cudaMemcpy(device_inputs,
                                   train_images.data.data() + offset,
                                   static_cast<size_t>(batch_size * input_dim) * sizeof(float),
                                   cudaMemcpyHostToDevice),
                       "copy inputs");
            check_cuda(cudaMemcpy(device_targets,
                                   train_labels.data.data() + label_offset,
                                   static_cast<size_t>(batch_size * num_classes) * sizeof(float),
                                   cudaMemcpyHostToDevice),
                       "copy targets");

            ForwardContext ctx;
            float* device_output = net.forward(device_inputs, batch_size, ctx);

            std::vector<float> host_output(static_cast<size_t>(batch_size * num_classes));
            check_cuda(cudaMemcpy(host_output.data(),
                                   device_output,
                                   host_output.size() * sizeof(float),
                                   cudaMemcpyDeviceToHost),
                       "read output");

            float loss = mse_loss(host_output, std::vector<float>(train_labels.data.data() + label_offset,
                                                                   train_labels.data.data() + label_offset +
                                                                       host_output.size()));
            epoch_loss += loss;
            epoch_acc += batch_accuracy(host_output,
                                        std::vector<float>(train_labels.data.data() + label_offset,
                                                           train_labels.data.data() + label_offset + host_output.size()),
                                        batch_size,
                                        num_classes);

            int threads = 256;
            int elements = batch_size * num_classes;
            int blocks = (elements + threads - 1) / threads;
            mse_grad_kernel<<<blocks, threads>>>(device_output, device_targets, device_grad_output, elements);

            net.backward(device_inputs, ctx, device_grad_output, device_grad_input, batch_size);
            net.adamw_update(learning_rate, beta1, beta2, eps, weight_decay);

            muon::NeuralNetwork::release_context(ctx);
        }

        std::cout << "Epoch " << (epoch + 1) << "/" << epochs
                  << " loss=" << epoch_loss / steps_per_epoch
                  << " acc=" << epoch_acc / steps_per_epoch << "\n";
    }

    // Quick evaluation on the first test batch.
    if (test_images.shape.size() != 2 || test_images.shape[1] != input_dim) {
        throw std::runtime_error("Test images must have shape [N, 784]");
    }
    if (test_labels.shape.size() != 2 || test_labels.shape[1] != num_classes) {
        throw std::runtime_error("Test labels must have shape [N, 10]");
    }

    const int test_samples = test_images.shape[0];
    if (test_samples == 0) {
        throw std::runtime_error("No test samples available for evaluation");
    }

    const int test_batch = std::min(test_samples, batch_size);
    check_cuda(cudaMemcpy(device_inputs,
                           test_images.data.data(),
                           static_cast<size_t>(test_batch * input_dim) * sizeof(float),
                           cudaMemcpyHostToDevice),
               "copy test inputs");
    check_cuda(cudaMemcpy(device_targets,
                           test_labels.data.data(),
                           static_cast<size_t>(test_batch * num_classes) * sizeof(float),
                           cudaMemcpyHostToDevice),
               "copy test targets");

    ForwardContext ctx;
    float* device_output = net.forward(device_inputs, test_batch, ctx);
    std::vector<float> host_output(static_cast<size_t>(test_batch * num_classes));
    check_cuda(cudaMemcpy(host_output.data(), device_output, host_output.size() * sizeof(float), cudaMemcpyDeviceToHost),
               "read test output");

    float test_loss = mse_loss(host_output, std::vector<float>(test_labels.data.begin(),
                                                               test_labels.data.begin() + host_output.size()));
    float test_acc = batch_accuracy(host_output, std::vector<float>(test_labels.data.begin(),
                                                                    test_labels.data.begin() + host_output.size()),
                                    test_batch,
                                    num_classes);

    std::cout << "Test batch loss=" << test_loss << " acc=" << test_acc << "\n";

    muon::NeuralNetwork::release_context(ctx);

    cudaFree(device_inputs);
    cudaFree(device_targets);
    cudaFree(device_grad_output);
    cudaFree(device_grad_input);

    return 0;
}

