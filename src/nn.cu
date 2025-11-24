#include "nn.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <fstream>
#include <random>
#include <stdexcept>

namespace muon {

namespace {

constexpr float kGeluCoeff = std::sqrt(2.0f / static_cast<float>(M_PI));

inline void check_cuda(cudaError_t err, const char* msg) {
    if (err != cudaSuccess) {
        throw std::runtime_error(std::string(msg) + ": " + cudaGetErrorString(err));
    }
}

__global__ void linear_forward_kernel(const float* x, const float* w, const float* b, float* y,
                                      int batch, int in_dim, int out_dim) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = batch * out_dim;
    if (idx >= total) return;
    int row = idx / out_dim;
    int col = idx - row * out_dim;
    float acc = b[col];
    const float* row_ptr = x + row * in_dim;
    const float* weight_row = w + col * in_dim;
    for (int k = 0; k < in_dim; ++k) {
        acc += row_ptr[k] * weight_row[k];
    }
    y[idx] = acc;
}

__global__ void gelu_forward_kernel(const float* x, float* y, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    float val = x[idx];
    y[idx] = 0.5f * val * (1.0f + erff(val / std::sqrt(2.0f)));
}

__global__ void gelu_backward_kernel(const float* pre_act, const float* grad_out, float* grad_in, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    float x = pre_act[idx];
    float erf_term = erff(x / std::sqrt(2.0f));
    float exp_term = expf(-0.5f * x * x);
    float grad = 0.5f * (1.0f + erf_term) + 0.5f * x * kGeluCoeff * exp_term;
    grad_in[idx] = grad_out[idx] * grad;
}

__global__ void linear_backward_input_kernel(const float* grad_out, const float* w, float* grad_in,
                                             int batch, int in_dim, int out_dim) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = batch * in_dim;
    if (idx >= total) return;
    int row = idx / in_dim;
    int col = idx - row * in_dim;
    float acc = 0.0f;
    const float* grad_row = grad_out + row * out_dim;
    for (int k = 0; k < out_dim; ++k) {
        acc += grad_row[k] * w[k * in_dim + col];
    }
    grad_in[idx] = acc;
}

__global__ void linear_backward_weight_kernel(const float* x, const float* grad_out, float* grad_w,
                                              int batch, int in_dim, int out_dim) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = out_dim * in_dim;
    if (idx >= total) return;
    int row = idx / in_dim;
    int col = idx - row * in_dim;
    float acc = 0.0f;
    const float* weight_grad = grad_out + row;
    for (int b = 0; b < batch; ++b) {
        acc += x[b * in_dim + col] * weight_grad[b * out_dim];
    }
    grad_w[idx] = acc;
}

__global__ void bias_backward_kernel(const float* grad_out, float* grad_b, int batch, int out_dim) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= out_dim) return;
    float acc = 0.0f;
    for (int b = 0; b < batch; ++b) {
        acc += grad_out[b * out_dim + idx];
    }
    grad_b[idx] = acc;
}

__global__ void sgd_kernel(float* param, const float* grad, float lr, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    param[idx] -= lr * grad[idx];
}

__global__ void zero_kernel(float* ptr, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) ptr[idx] = 0.0f;
}

}  // namespace

NeuralNetwork::NeuralNetwork(const std::vector<int>& layer_sizes) : layer_sizes_(layer_sizes) {
    if (layer_sizes_.size() < 2) {
        throw std::invalid_argument("Network needs at least input and output size");
    }
    allocate_layers();
    initialize_parameters();
}

NeuralNetwork::~NeuralNetwork() { release(); }

NeuralNetwork::NeuralNetwork(NeuralNetwork&& other) noexcept { *this = std::move(other); }

NeuralNetwork& NeuralNetwork::operator=(NeuralNetwork&& other) noexcept {
    if (this == &other) return *this;
    release();
    layer_sizes_ = std::move(other.layer_sizes_);
    layers_ = std::move(other.layers_);
    other.layers_.clear();
    return *this;
}

void NeuralNetwork::allocate_layers() {
    layers_.resize(layer_sizes_.size() - 1);
    for (size_t i = 0; i + 1 < layer_sizes_.size(); ++i) {
        LinearLayer layer;
        layer.in_dim = layer_sizes_[i];
        layer.out_dim = layer_sizes_[i + 1];

        size_t weight_size = static_cast<size_t>(layer.in_dim) * layer.out_dim;
        check_cuda(cudaMalloc(&layer.weights, weight_size * sizeof(float)), "malloc weights");
        check_cuda(cudaMalloc(&layer.bias, layer.out_dim * sizeof(float)), "malloc bias");
        check_cuda(cudaMalloc(&layer.grad_weights, weight_size * sizeof(float)), "malloc grad weights");
        check_cuda(cudaMalloc(&layer.grad_bias, layer.out_dim * sizeof(float)), "malloc grad bias");
        layers_[i] = layer;
    }
}

void NeuralNetwork::initialize_parameters() {
    std::mt19937 gen(42);
    for (auto& layer : layers_) {
        float scale = std::sqrt(2.0f / static_cast<float>(layer.in_dim + layer.out_dim));
        std::normal_distribution<float> dist(0.0f, scale);
        size_t weight_size = static_cast<size_t>(layer.in_dim) * layer.out_dim;
        std::vector<float> host_weights(weight_size);
        std::vector<float> host_bias(layer.out_dim);
        for (size_t i = 0; i < weight_size; ++i) host_weights[i] = dist(gen);
        for (int i = 0; i < layer.out_dim; ++i) host_bias[i] = 0.0f;
        check_cuda(cudaMemcpy(layer.weights, host_weights.data(), weight_size * sizeof(float), cudaMemcpyHostToDevice),
                   "copy weights");
        check_cuda(cudaMemcpy(layer.bias, host_bias.data(), layer.out_dim * sizeof(float), cudaMemcpyHostToDevice),
                   "copy bias");
    }
}

float* NeuralNetwork::forward(const float* device_input, int batch_size, ForwardContext& ctx, cudaStream_t stream) const {
    if (batch_size <= 0) {
        throw std::invalid_argument("Batch size must be positive");
    }
    ctx.caches.resize(layers_.size());

    const float* current = device_input;
    float* output = nullptr;
    for (size_t i = 0; i < layers_.size(); ++i) {
        const auto& layer = layers_[i];
        size_t pre_size = static_cast<size_t>(batch_size) * layer.out_dim;
        float* pre_act = nullptr;
        float* act = nullptr;
        check_cuda(cudaMalloc(&pre_act, pre_size * sizeof(float)), "alloc pre activation");
        check_cuda(cudaMalloc(&act, pre_size * sizeof(float)), "alloc activation");

        int threads = 256;
        int blocks_linear = static_cast<int>((pre_size + threads - 1) / threads);
        linear_forward_kernel<<<blocks_linear, threads, 0, stream>>>(current, layer.weights, layer.bias, pre_act,
                                                                    batch_size, layer.in_dim, layer.out_dim);
        gelu_forward_kernel<<<blocks_linear, threads, 0, stream>>>(pre_act, act, static_cast<int>(pre_size));

        ctx.caches[i] = {pre_act, act};
        current = act;
        output = act;
    }
    return output;
}

void NeuralNetwork::backward(const float* device_input,
                             const ForwardContext& ctx,
                             const float* grad_output,
                             float* grad_input,
                             int batch_size,
                             cudaStream_t stream) {
    if (static_cast<int>(ctx.caches.size()) != num_layers()) {
        throw std::invalid_argument("Forward context does not match network layers");
    }

    const float* upstream = grad_output;
    float* grad_current_input = nullptr;

    for (int layer_idx = static_cast<int>(layers_.size()) - 1; layer_idx >= 0; --layer_idx) {
        const auto& layer = layers_[layer_idx];
        const auto& cache = ctx.caches[layer_idx];
        size_t elem = static_cast<size_t>(batch_size) * layer.out_dim;

        float* grad_activation = nullptr;
        check_cuda(cudaMalloc(&grad_activation, elem * sizeof(float)), "alloc grad activation");

        int threads = 256;
        int blocks_elem = static_cast<int>((elem + threads - 1) / threads);
        gelu_backward_kernel<<<blocks_elem, threads, 0, stream>>>(cache.pre_activation, upstream, grad_activation,
                                                                  static_cast<int>(elem));

        // Gradients w.r.t. parameters and inputs.
        size_t grad_w_size = static_cast<size_t>(layer.in_dim) * layer.out_dim;
        int blocks_weights = static_cast<int>((grad_w_size + threads - 1) / threads);
        int blocks_bias = static_cast<int>((layer.out_dim + threads - 1) / threads);
        zero_kernel<<<blocks_weights, threads, 0, stream>>>(layer.grad_weights, static_cast<int>(grad_w_size));
        zero_kernel<<<blocks_bias, threads, 0, stream>>>(layer.grad_bias, layer.out_dim);

        linear_backward_weight_kernel<<<blocks_weights, threads, 0, stream>>>(
            layer_idx == 0 ? device_input : ctx.caches[layer_idx - 1].activation, grad_activation, layer.grad_weights,
            batch_size, layer.in_dim, layer.out_dim);
        bias_backward_kernel<<<blocks_bias, threads, 0, stream>>>(grad_activation, layer.grad_bias, batch_size,
                                                                  layer.out_dim);

        // Gradient for previous activation.
        size_t grad_in_size = static_cast<size_t>(batch_size) * layer.in_dim;
        int blocks_in = static_cast<int>((grad_in_size + threads - 1) / threads);
        check_cuda(cudaMalloc(&grad_current_input, grad_in_size * sizeof(float)), "alloc grad input");
        linear_backward_input_kernel<<<blocks_in, threads, 0, stream>>>(grad_activation, layer.weights,
                                                                        grad_current_input, batch_size, layer.in_dim,
                                                                        layer.out_dim);

        check_cuda(cudaFree(grad_activation), "free grad activation");
        upstream = grad_current_input;
    }

    // Copy gradient at input to caller buffer.
    size_t input_elems = static_cast<size_t>(batch_size) * layer_sizes_.front();
    check_cuda(cudaMemcpyAsync(grad_input, grad_current_input, input_elems * sizeof(float), cudaMemcpyDeviceToDevice,
                               stream),
               "copy grad input");
    check_cuda(cudaFree(grad_current_input), "free grad input");
}

void NeuralNetwork::sgd_update(float learning_rate, cudaStream_t stream) {
    int threads = 256;
    for (auto& layer : layers_) {
        size_t w_elems = static_cast<size_t>(layer.in_dim) * layer.out_dim;
        int blocks_w = static_cast<int>((w_elems + threads - 1) / threads);
        int blocks_b = static_cast<int>((layer.out_dim + threads - 1) / threads);
        sgd_kernel<<<blocks_w, threads, 0, stream>>>(layer.weights, layer.grad_weights, learning_rate,
                                                     static_cast<int>(w_elems));
        sgd_kernel<<<blocks_b, threads, 0, stream>>>(layer.bias, layer.grad_bias, learning_rate, layer.out_dim);
    }
}

void NeuralNetwork::release_context(ForwardContext& ctx) {
    for (auto& cache : ctx.caches) {
        if (cache.pre_activation) cudaFree(cache.pre_activation);
        if (cache.activation) cudaFree(cache.activation);
    }
    ctx.caches.clear();
}

void NeuralNetwork::save(const std::string& path) const {
    std::ofstream out(path, std::ios::binary);
    if (!out) {
        throw std::runtime_error("Failed to open file for writing: " + path);
    }
    const uint32_t magic = 0x4d554f4e;  // 'MUON'
    uint32_t version = 1;
    uint32_t num_layers_u32 = static_cast<uint32_t>(layer_sizes_.size());
    out.write(reinterpret_cast<const char*>(&magic), sizeof(magic));
    out.write(reinterpret_cast<const char*>(&version), sizeof(version));
    out.write(reinterpret_cast<const char*>(&num_layers_u32), sizeof(num_layers_u32));
    out.write(reinterpret_cast<const char*>(layer_sizes_.data()), layer_sizes_.size() * sizeof(int));

    for (const auto& layer : layers_) {
        size_t weight_size = static_cast<size_t>(layer.in_dim) * layer.out_dim;
        std::vector<float> host_weights(weight_size);
        std::vector<float> host_bias(layer.out_dim);
        check_cuda(cudaMemcpy(host_weights.data(), layer.weights, weight_size * sizeof(float), cudaMemcpyDeviceToHost),
                   "read weights for save");
        check_cuda(cudaMemcpy(host_bias.data(), layer.bias, layer.out_dim * sizeof(float), cudaMemcpyDeviceToHost),
                   "read bias for save");
        out.write(reinterpret_cast<const char*>(host_weights.data()), weight_size * sizeof(float));
        out.write(reinterpret_cast<const char*>(host_bias.data()), layer.out_dim * sizeof(float));
    }
}

void NeuralNetwork::load(const std::string& path) {
    std::ifstream in(path, std::ios::binary);
    if (!in) {
        throw std::runtime_error("Failed to open file for reading: " + path);
    }
    uint32_t magic = 0;
    uint32_t version = 0;
    uint32_t num_layers_u32 = 0;
    in.read(reinterpret_cast<char*>(&magic), sizeof(magic));
    in.read(reinterpret_cast<char*>(&version), sizeof(version));
    in.read(reinterpret_cast<char*>(&num_layers_u32), sizeof(num_layers_u32));
    if (magic != 0x4d554f4e) {
        throw std::runtime_error("Invalid model file magic");
    }
    if (version != 1) {
        throw std::runtime_error("Unsupported model version");
    }
    std::vector<int> sizes(num_layers_u32);
    in.read(reinterpret_cast<char*>(sizes.data()), sizes.size() * sizeof(int));
    layer_sizes_ = sizes;
    release();
    allocate_layers();

    for (auto& layer : layers_) {
        size_t weight_size = static_cast<size_t>(layer.in_dim) * layer.out_dim;
        std::vector<float> host_weights(weight_size);
        std::vector<float> host_bias(layer.out_dim);
        in.read(reinterpret_cast<char*>(host_weights.data()), weight_size * sizeof(float));
        in.read(reinterpret_cast<char*>(host_bias.data()), layer.out_dim * sizeof(float));
        check_cuda(cudaMemcpy(layer.weights, host_weights.data(), weight_size * sizeof(float), cudaMemcpyHostToDevice),
                   "load weights");
        check_cuda(cudaMemcpy(layer.bias, host_bias.data(), layer.out_dim * sizeof(float), cudaMemcpyHostToDevice),
                   "load bias");
    }
}

void NeuralNetwork::release() {
    for (auto& layer : layers_) {
        if (layer.weights) cudaFree(layer.weights);
        if (layer.bias) cudaFree(layer.bias);
        if (layer.grad_weights) cudaFree(layer.grad_weights);
        if (layer.grad_bias) cudaFree(layer.grad_bias);
    }
    layers_.clear();
}

}  // namespace muon
