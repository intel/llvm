// REQUIRES: aspect-usm_shared_allocations
// RUN: %{build} -o %t.out
// RUN: %{run} %t.out

// Tests the enqueue free function kernel shortcuts with sycl::kernel.
// NOTE: This relies on the availability of an OpenCL C compiler being
// available.

#include <sycl/detail/core.hpp>
#include <sycl/ext/oneapi/experimental/enqueue_functions.hpp>
#include <sycl/usm.hpp>

#include "common.hpp"

#include <iostream>

namespace oneapiext = sycl::ext::oneapi::experimental;

constexpr size_t N = 1024;

int main() {
  sycl::queue Q;

  if (!Q.get_device().ext_oneapi_can_compile(
          oneapiext::source_language::opencl)) {
    std::cout
        << "Backend does not support OpenCL C source kernel bundle extension: "
        << Q.get_backend() << std::endl;
    return 0;
  }

  auto KB = CreateKB(Q);

  assert(KB.ext_oneapi_has_kernel("KernelSingleTask"));
  assert(KB.ext_oneapi_has_kernel("Kernel1D"));
  assert(KB.ext_oneapi_has_kernel("Kernel2D"));
  assert(KB.ext_oneapi_has_kernel("Kernel3D"));

  sycl::kernel KernelSingleTask = KB.ext_oneapi_get_kernel("KernelSingleTask");
  sycl::kernel Kernel1D = KB.ext_oneapi_get_kernel("Kernel1D");
  sycl::kernel Kernel2D = KB.ext_oneapi_get_kernel("Kernel2D");
  sycl::kernel Kernel3D = KB.ext_oneapi_get_kernel("Kernel3D");

  int *Memory = sycl::malloc_shared<int>(N, Q);

  int Failed = 0;

  // single_task shortcut
  oneapiext::single_task(Q, KernelSingleTask, (int)N, 42, Memory);
  Q.wait();
  for (size_t I = 0; I < N; ++I)
    Failed += Check(Memory, 42, I, "single_task shortcut");

  // 1D parallel_for shortcut
  oneapiext::parallel_for(Q, sycl::range<1>{N}, Kernel1D, 43, Memory);
  Q.wait();
  for (size_t I = 0; I < N; ++I)
    Failed += Check(Memory, 43, I, "1D parallel_for shortcut");

  // 2D parallel_for shortcut
  oneapiext::parallel_for(Q, sycl::range<2>{8, N / 8}, Kernel2D, 44, Memory);
  Q.wait();
  for (size_t I = 0; I < N; ++I)
    Failed += Check(Memory, 44, I, "2D parallel_for shortcut");

  // 3D parallel_for shortcut
  oneapiext::parallel_for(Q, sycl::range<3>{8, 8, N / 64}, Kernel3D, 45,
                          Memory);
  Q.wait();
  for (size_t I = 0; I < N; ++I)
    Failed += Check(Memory, 45, I, "3D parallel_for shortcut");

  // 1D nd_launch shortcut
  oneapiext::nd_launch(Q, sycl::nd_range<1>{sycl::range<1>{N}, sycl::range{8}},
                       Kernel1D, 46, Memory);
  Q.wait();
  for (size_t I = 0; I < N; ++I)
    Failed += Check(Memory, 46, I, "1D nd_launch shortcut");

  // 2D nd_launch shortcut
  oneapiext::nd_launch(
      Q, sycl::nd_range<2>{sycl::range<2>{8, N / 8}, sycl::range{8, 8}},
      Kernel2D, 47, Memory);
  Q.wait();
  for (size_t I = 0; I < N; ++I)
    Failed += Check(Memory, 47, I, "2D nd_launch shortcut");

  // 3D nd_launch shortcut
  oneapiext::nd_launch(
      Q, sycl::nd_range<3>{sycl::range<3>{8, 8, N / 64}, sycl::range{8, 8, 8}},
      Kernel3D, 48, Memory);
  Q.wait();
  for (size_t I = 0; I < N; ++I)
    Failed += Check(Memory, 48, I, "3D nd_launch shortcut");

  // 1D parallel_for shortcut with launch config
  oneapiext::parallel_for(
      Q, oneapiext::launch_config<sycl::range<1>>{sycl::range<1>{N}}, Kernel1D,
      49, Memory);
  Q.wait();
  for (size_t I = 0; I < N; ++I)
    Failed +=
        Check(Memory, 49, I, "1D parallel_for shortcut with launch config");

  // 2D parallel_for shortcut with launch config
  oneapiext::parallel_for(
      Q, oneapiext::launch_config<sycl::range<2>>{sycl::range<2>{8, N / 8}},
      Kernel2D, 50, Memory);
  Q.wait();
  for (size_t I = 0; I < N; ++I)
    Failed +=
        Check(Memory, 50, I, "2D parallel_for shortcut with launch config");

  // 3D parallel_for shortcut with launch config
  oneapiext::parallel_for(
      Q, oneapiext::launch_config<sycl::range<3>>{sycl::range<3>{8, 8, N / 64}},
      Kernel3D, 51, Memory);
  Q.wait();
  for (size_t I = 0; I < N; ++I)
    Failed +=
        Check(Memory, 51, I, "3D parallel_for shortcut with launch config");

  // 1D nd_launch shortcut with launch config
  oneapiext::nd_launch(
      Q,
      oneapiext::launch_config<sycl::nd_range<1>>{
          sycl::nd_range<1>{sycl::range<1>{N}, sycl::range{8}}},
      Kernel1D, 52, Memory);
  Q.wait();
  for (size_t I = 0; I < N; ++I)
    Failed += Check(Memory, 52, I, "1D nd_launch shortcut with launch config");

  // 2D nd_launch shortcut with launch config
  oneapiext::nd_launch(
      Q,
      oneapiext::launch_config<sycl::nd_range<2>>{
          sycl::nd_range<2>{sycl::range<2>{8, N / 8}, sycl::range{8, 8}}},
      Kernel2D, 53, Memory);
  Q.wait();
  for (size_t I = 0; I < N; ++I)
    Failed += Check(Memory, 53, I, "2D nd_launch shortcut with launch config");

  // 3D nd_launch shortcut with launch config
  oneapiext::nd_launch(
      Q,
      oneapiext::launch_config<sycl::nd_range<3>>{sycl::nd_range<3>{
          sycl::range<3>{8, 8, N / 64}, sycl::range{8, 8, 8}}},
      Kernel3D, 54, Memory);
  Q.wait();
  for (size_t I = 0; I < N; ++I)
    Failed += Check(Memory, 54, I, "3D nd_launch shortcut with launch config");

  sycl::free(Memory, Q);
  return Failed;
}
