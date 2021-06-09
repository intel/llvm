// RUN: %clangxx -fsycl -DUSE_OPENCL %s
// RUN: %clangxx -fsycl -DUSE_L0 %s
// RUN: %clangxx -fsycl -DUSE_CUDA %s

#ifdef USE_OPENCL
#include <CL/cl.h>

#include <CL/sycl/backend/opencl.hpp>

constexpr auto Backend = sycl::backend::opencl;
#endif

#ifdef USE_L0
#include <level_zero/ze_api.h>

#include <CL/sycl/backend/level_zero.hpp>

constexpr auto Backend = sycl::backend::level_zero;
#endif

#ifdef USE_CUDA
#include <CL/sycl/backend/cuda.hpp>

constexpr auto Backend = sycl::backend::cuda;
#endif

#include <sycl/sycl.hpp>

int main() {
#ifdef USE_OPENCL
  static_assert(
      std::is_same_v<sycl::backend_traits<Backend>::input_type<sycl::device>,
                     sycl::interop<Backend, sycl::device>::type>);
  static_assert(
      std::is_same_v<sycl::backend_traits<Backend>::input_type<sycl::context>,
                     sycl::interop<Backend, sycl::context>::type>);
  static_assert(
      std::is_same_v<sycl::backend_traits<Backend>::input_type<sycl::queue>,
                     sycl::interop<Backend, sycl::queue>::type>);
  static_assert(std::is_same_v<
                sycl::backend_traits<Backend>::input_type<sycl::buffer<int, 2>>,
                sycl::interop<Backend, sycl::buffer<int, 2>>::type>);
  static_assert(
      std::is_same_v<sycl::backend_traits<Backend>::input_type<sycl::event>,
                     sycl::interop<Backend, sycl::event>::type>);
#endif

// CUDA does not have a native type for platforms
#ifndef USE_CUDA
  static_assert(
      std::is_same_v<sycl::backend_traits<Backend>::input_type<sycl::platform>,
                     sycl::interop<Backend, sycl::platform>::type>);
  static_assert(
      std::is_same_v<sycl::backend_traits<Backend>::return_type<sycl::platform>,
                     sycl::interop<Backend, sycl::platform>::type>);
#endif

  static_assert(
      std::is_same_v<sycl::backend_traits<Backend>::return_type<sycl::device>,
                     sycl::interop<Backend, sycl::device>::type>);
  static_assert(
      std::is_same_v<sycl::backend_traits<Backend>::return_type<sycl::context>,
                     sycl::interop<Backend, sycl::context>::type>);
  static_assert(
      std::is_same_v<sycl::backend_traits<Backend>::return_type<sycl::queue>,
                     sycl::interop<Backend, sycl::queue>::type>);

  return 0;
}
