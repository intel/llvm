// RUN: %clangxx -fsycl -DUSE_OPENCL %s
// RUN: %clangxx -fsycl -DUSE_L0 %s
// RUN: %clangxx -fsycl -DUSE_CUDA %s
// RUN: %clangxx -fsycl -DUSE_CUDA_EXPERIMENTAL %s

#ifdef USE_OPENCL
#include <CL/cl.h>

#include <CL/sycl/backend/opencl.hpp>

constexpr auto Backend = sycl::backend::opencl;
#endif

#ifdef USE_L0
#include <level_zero/ze_api.h>

#include <sycl/ext/oneapi/backend/level_zero.hpp>

constexpr auto Backend = sycl::backend::ext_oneapi_level_zero;
#endif

#ifdef USE_CUDA
#include <CL/sycl/detail/backend_traits_cuda.hpp>

constexpr auto Backend = sycl::backend::ext_oneapi_cuda;
#endif

#ifdef USE_CUDA_EXPERIMENTAL
#define SYCL_EXT_ONEAPI_BACKEND_CUDA_EXPERIMENTAL 1
#include <sycl/sycl.hpp>
#include <sycl/ext/oneapi/experimental/backend/cuda.hpp>

constexpr auto Backend = sycl::backend::ext_oneapi_cuda;
#endif

#include <sycl/sycl.hpp>

int main() {
#ifdef USE_OPENCL
  static_assert(
      std::is_same_v<sycl::backend_traits<Backend>::input_type<sycl::device>,
                     sycl::detail::interop<Backend, sycl::device>::type>);
  static_assert(
      std::is_same_v<sycl::backend_traits<Backend>::input_type<sycl::context>,
                     sycl::detail::interop<Backend, sycl::context>::type>);
  static_assert(
      std::is_same_v<sycl::backend_traits<Backend>::input_type<sycl::queue>,
                     sycl::detail::interop<Backend, sycl::queue>::type>);
  static_assert(
      std::is_same_v<sycl::backend_traits<Backend>::input_type<sycl::event>,
                     sycl::detail::interop<Backend, sycl::event>::type>);
#endif

// CUDA does not have a native type for platforms
#ifndef USE_CUDA
  static_assert(
      std::is_same_v<sycl::backend_traits<Backend>::input_type<sycl::platform>,
                     sycl::detail::interop<Backend, sycl::platform>::type>);
  static_assert(
      std::is_same_v<sycl::backend_traits<Backend>::return_type<sycl::platform>,
                     sycl::detail::interop<Backend, sycl::platform>::type>);
#endif

  static_assert(
      std::is_same_v<sycl::backend_traits<Backend>::return_type<sycl::device>,
                     sycl::detail::interop<Backend, sycl::device>::type>);

// CUDA experimental return type is different to inpt type
#ifndef USE_CUDA_EXPERIMENTAL
  static_assert(
      std::is_same_v<sycl::backend_traits<Backend>::return_type<sycl::context>,
                     sycl::detail::interop<Backend, sycl::context>::type>);
#endif
  static_assert(
      std::is_same_v<sycl::backend_traits<Backend>::return_type<sycl::queue>,
                     sycl::detail::interop<Backend, sycl::queue>::type>);

  return 0;
}
