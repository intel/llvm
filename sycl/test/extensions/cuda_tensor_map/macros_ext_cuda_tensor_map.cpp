// RUN: %clangxx -fsycl -fsyntax-only %s
#include <sycl/ext/codeplay/experimental/cuda_tensor_map.hpp>

#ifndef SYCL_EXT_CODEPLAY_CUDA_TENSOR_MAP
#error SYCL_EXT_CODEPLAY_CUDA_TENSOR_MAP is not defined
#endif
#if SYCL_EXT_CODEPLAY_CUDA_TENSOR_MAP != 1
#error SYCL_EXT_CODEPLAY_CUDA_TENSOR_MAP has unexpected value
#endif
