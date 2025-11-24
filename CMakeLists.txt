cmake_minimum_required(VERSION 3.18)
project(muon_nn LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

add_library(muon_nn STATIC src/nn.cu)
target_include_directories(muon_nn PUBLIC ${PROJECT_SOURCE_DIR}/include)
set_target_properties(muon_nn PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

add_executable(xor_example examples/xor.cu)
target_link_libraries(xor_example PRIVATE muon_nn)
set_target_properties(xor_example PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

add_executable(mnist_example examples/mnist.cu)
target_link_libraries(mnist_example PRIVATE muon_nn)
set_target_properties(mnist_example PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

add_executable(cifar100_example examples/cifar100.cu)
target_link_libraries(cifar100_example PRIVATE muon_nn)
set_target_properties(cifar100_example PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

add_executable(image_classifier tools/image_classifier.cpp)
