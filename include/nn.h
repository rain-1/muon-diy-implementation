#pragma once

#include <cuda_runtime.h>
#include <cstdint>
#include <string>
#include <vector>

namespace muon {

struct LayerCache {
    float* pre_activation{nullptr};
    float* activation{nullptr};
};

struct LinearLayer {
    int in_dim{0};
    int out_dim{0};
    float* weights{nullptr};
    float* bias{nullptr};
    float* grad_weights{nullptr};
    float* grad_bias{nullptr};
    // Adam moment estimates.
    float* m_weights{nullptr};
    float* v_weights{nullptr};
    float* m_bias{nullptr};
    float* v_bias{nullptr};
};

struct ForwardContext {
    std::vector<LayerCache> caches;
};

class NeuralNetwork {
  public:
    NeuralNetwork() = default;
    explicit NeuralNetwork(const std::vector<int>& layer_sizes);
    ~NeuralNetwork();

    NeuralNetwork(const NeuralNetwork&) = delete;
    NeuralNetwork& operator=(const NeuralNetwork&) = delete;

    NeuralNetwork(NeuralNetwork&& other) noexcept;
    NeuralNetwork& operator=(NeuralNetwork&& other) noexcept;

    int num_layers() const { return static_cast<int>(layers_.size()); }
    const std::vector<int>& layer_sizes() const { return layer_sizes_; }

    // Forward pass. Input/outputs are device pointers in row-major [batch, dim].
    // The context stores per-layer activations required for backward.
    float* forward(const float* device_input, int batch_size, ForwardContext& ctx, cudaStream_t stream = 0) const;

    // Backward pass. grad_output points to gradient at network output.
    // Writes gradient at the network input into grad_input (device pointer sized batch x input_dim).
    void backward(const float* device_input,
                  const ForwardContext& ctx,
                  const float* grad_output,
                  float* grad_input,
                  int batch_size,
                  cudaStream_t stream = 0);

    // Apply a simple SGD update to parameters with the given learning rate.
    void sgd_update(float learning_rate, cudaStream_t stream = 0);

    // Apply AdamW update with bias correction and weight decay.
    void adamw_update(float learning_rate,
                      float beta1 = 0.9f,
                      float beta2 = 0.999f,
                      float eps = 1e-8f,
                      float weight_decay = 0.01f,
                      cudaStream_t stream = 0);

    // Release device memory allocated during forward passes.
    static void release_context(ForwardContext& ctx);

    // Serialize network metadata and parameters to binary file.
    void save(const std::string& path) const;

    // Load network from binary file, replacing existing parameters.
    void load(const std::string& path);

  private:
    std::vector<int> layer_sizes_;
    std::vector<LinearLayer> layers_;
    int64_t adam_step_{0};

    void allocate_layers();
    void initialize_parameters();
    void release();
};

}  // namespace muon
