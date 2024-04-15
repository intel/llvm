// REQUIRES: cuda
// RUN: %clangxx -fsycl -fsyntax-only %s
// RUN: %clangxx -fsycl -fsyntax-only -DUSE_CUDA_EXPERIMENTAL %s

#ifdef USE_CUDA_EXPERIMENTAL
#define SYCL_EXT_ONEAPI_BACKEND_CUDA 1
#define SYCL_EXT_ONEAPI_BACKEND_CUDA_EXPERIMENTAL 1
#include <sycl/ext/oneapi/experimental/backend/cuda.hpp>
#endif

#include <sycl/sycl.hpp>

constexpr auto Backend = sycl::backend::ext_oneapi_cuda;

int main() {
  static_assert(
      std::is_same_v<sycl::backend_traits<Backend>::return_type<sycl::device>,
                     sycl::detail::interop<Backend, sycl::device>::type>);
#ifndef USE_CUDA_EXPERIMENTAL
  // CUDA experimental return type is different to input type
  static_assert(
      std::is_same_v<sycl::backend_traits<Backend>::return_type<sycl::context>,
                     sycl::detail::interop<Backend, sycl::context>::type>);
#endif
  static_assert(
      std::is_same_v<sycl::backend_traits<Backend>::return_type<sycl::queue>,
                     sycl::detail::interop<Backend, sycl::queue>::type>);

  return 0;
}
