#include "nn.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <fstream>
#include <iostream>
#include <numeric>
#include <random>
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

std::vector<float> softmax(const std::vector<float>& logits, int batch, int classes) {
    std::vector<float> probs(logits.size());
    for (int i = 0; i < batch; ++i) {
        const float* row = logits.data() + static_cast<size_t>(i) * classes;
        float row_max = row[0];
        for (int c = 1; c < classes; ++c) row_max = std::max(row_max, row[c]);

        float sum = 0.0f;
        for (int c = 0; c < classes; ++c) {
            float e = std::exp(row[c] - row_max);
            probs[static_cast<size_t>(i) * classes + c] = e;
            sum += e;
        }
        float inv_sum = 1.0f / sum;
        for (int c = 0; c < classes; ++c) {
            probs[static_cast<size_t>(i) * classes + c] *= inv_sum;
        }
    }
    return probs;
}

float cross_entropy(const std::vector<float>& probs, const std::vector<float>& targets, int batch, int classes) {
    float loss = 0.0f;
    for (int i = 0; i < batch; ++i) {
        int target_idx = 0;
        for (int c = 0; c < classes; ++c) {
            if (targets[static_cast<size_t>(i) * classes + c] > targets[static_cast<size_t>(i) * classes + target_idx]) {
                target_idx = c;
            }
        }
        float prob = probs[static_cast<size_t>(i) * classes + target_idx];
        loss -= std::log(std::max(prob, 1e-10f));
    }
    return loss / static_cast<float>(batch);
}

}  // namespace

int main() {
    const std::string dataset_root = "datasets/cifar100/processed";
    const std::string train_images_path = dataset_root + "/train_images.bin";
    const std::string train_labels_path = dataset_root + "/train_labels.bin";
    const std::string test_images_path = dataset_root + "/test_images.bin";
    const std::string test_labels_path = dataset_root + "/test_labels.bin";

    Tensor train_images = load_tensor(train_images_path);
    Tensor train_labels = load_tensor(train_labels_path);
    Tensor test_images = load_tensor(test_images_path);
    Tensor test_labels = load_tensor(test_labels_path);

    if (train_images.shape.size() != 2 || train_images.shape[1] != 32 * 32 * 3) {
        throw std::runtime_error("Train images must have shape [N, 3072]");
    }
    if (train_labels.shape.size() != 2 || train_labels.shape[1] != 100) {
        throw std::runtime_error("Train labels must have shape [N, 100]");
    }

    const int train_samples = train_images.shape[0];
    const int input_dim = train_images.shape[1];
    const int num_classes = train_labels.shape[1];

    NeuralNetwork net({input_dim, 512, 256, 128, num_classes});

    const int batch_size = 128;
    const int epochs = 30;
    const float learning_rate = 0.001f;
    const float beta1 = 0.9f;
    const float beta2 = 0.999f;
    const float eps = 1e-8f;
    const float weight_decay = 0.005f;
    const int steps_per_epoch = train_samples / batch_size;

    float* device_inputs = nullptr;
    float* device_grad_output = nullptr;
    float* device_grad_input = nullptr;

    check_cuda(cudaMalloc(&device_inputs, static_cast<size_t>(batch_size * input_dim) * sizeof(float)), "alloc inputs");
    check_cuda(cudaMalloc(&device_grad_output, static_cast<size_t>(batch_size * num_classes) * sizeof(float)),
               "alloc grad output");
    check_cuda(cudaMalloc(&device_grad_input, static_cast<size_t>(batch_size * input_dim) * sizeof(float)),
               "alloc grad input");

    std::vector<float> host_batch_inputs(static_cast<size_t>(batch_size * input_dim));
    std::vector<float> host_batch_labels(static_cast<size_t>(batch_size * num_classes));
    std::vector<int> indices(train_samples);
    std::iota(indices.begin(), indices.end(), 0);

    std::mt19937 rng(42);

    for (int epoch = 0; epoch < epochs; ++epoch) {
        float epoch_loss = 0.0f;
        float epoch_acc = 0.0f;

        std::vector<float> host_output(static_cast<size_t>(batch_size * num_classes));
        std::vector<float> host_grad_output(host_output.size());

        std::shuffle(indices.begin(), indices.end(), rng);

        for (int step = 0; step < steps_per_epoch; ++step) {
            for (int b = 0; b < batch_size; ++b) {
                int idx = indices[step * batch_size + b];
                std::copy_n(train_images.data.data() + static_cast<size_t>(idx) * input_dim,
                            input_dim,
                            host_batch_inputs.data() + static_cast<size_t>(b) * input_dim);
                std::copy_n(train_labels.data.data() + static_cast<size_t>(idx) * num_classes,
                            num_classes,
                            host_batch_labels.data() + static_cast<size_t>(b) * num_classes);
            }

            check_cuda(cudaMemcpy(device_inputs,
                                   host_batch_inputs.data(),
                                   static_cast<size_t>(batch_size * input_dim) * sizeof(float),
                                   cudaMemcpyHostToDevice),
                       "copy inputs");

            ForwardContext ctx;
            float* device_output = net.forward(device_inputs, batch_size, ctx);

            check_cuda(cudaMemcpy(host_output.data(),
                                   device_output,
                                   host_output.size() * sizeof(float),
                                   cudaMemcpyDeviceToHost),
                       "read output");

            std::vector<float> probs = softmax(host_output, batch_size, num_classes);
            epoch_loss += cross_entropy(probs, host_batch_labels, batch_size, num_classes);
            epoch_acc += batch_accuracy(probs, host_batch_labels, batch_size, num_classes);

            for (size_t i = 0; i < host_grad_output.size(); ++i) {
                host_grad_output[i] = (probs[i] - host_batch_labels[i]) / static_cast<float>(batch_size);
            }
            check_cuda(cudaMemcpy(device_grad_output,
                                   host_grad_output.data(),
                                   host_grad_output.size() * sizeof(float),
                                   cudaMemcpyHostToDevice),
                       "copy grad output");

            net.backward(device_inputs, ctx, device_grad_output, device_grad_input, batch_size);
            net.adamw_update(learning_rate, beta1, beta2, eps, weight_decay);

            muon::NeuralNetwork::release_context(ctx);
        }

        std::cout << "Epoch " << (epoch + 1) << "/" << epochs
                  << " loss=" << epoch_loss / steps_per_epoch
                  << " acc=" << epoch_acc / steps_per_epoch << "\n";
    }

    if (test_images.shape.size() != 2 || test_images.shape[1] != input_dim) {
        throw std::runtime_error("Test images must have shape [N, 3072]");
    }
    if (test_labels.shape.size() != 2 || test_labels.shape[1] != num_classes) {
        throw std::runtime_error("Test labels must have shape [N, 100]");
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

    ForwardContext ctx;
    float* device_output = net.forward(device_inputs, test_batch, ctx);
    std::vector<float> host_output(static_cast<size_t>(test_batch * num_classes));
    check_cuda(cudaMemcpy(host_output.data(), device_output, host_output.size() * sizeof(float), cudaMemcpyDeviceToHost),
               "read test output");

    std::vector<float> test_targets(test_labels.data.begin(), test_labels.data.begin() + host_output.size());
    std::vector<float> test_probs = softmax(host_output, test_batch, num_classes);
    float test_loss = cross_entropy(test_probs, test_targets, test_batch, num_classes);
    float test_acc = batch_accuracy(test_probs, test_targets, test_batch, num_classes);

    std::cout << "Test batch loss=" << test_loss << " acc=" << test_acc << "\n";

    muon::NeuralNetwork::release_context(ctx);

    cudaFree(device_inputs);
    cudaFree(device_grad_output);
    cudaFree(device_grad_input);

    return 0;
}
