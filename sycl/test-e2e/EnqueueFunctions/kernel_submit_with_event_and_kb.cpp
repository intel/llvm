// RUN: %{build} -o %t.out
// RUN: %{run} %t.out

// Tests the enqueue free function kernel submit_with_events with events and
// sycl::kernel. NOTE: This relies on the availability of an OpenCL C compiler
// being available.

#include <sycl/detail/core.hpp>
#include <sycl/ext/oneapi/experimental/enqueue_functions.hpp>
#include <sycl/usm.hpp>

#include "common.hpp"

#include <iostream>

namespace oneapiext = sycl::ext::oneapi::experimental;

constexpr size_t N = 1024;

int main() {
  sycl::queue Q;
  int Memory[N] = {0};

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

  int Failed = 0;

  // single_task
  {
    sycl::buffer<int, 1> MemBuf{Memory, sycl::range<1>{N}};
    {
      oneapiext::submit_with_event(Q, [&](sycl::handler &CGH) {
        sycl::accessor MemAcc{MemBuf, CGH, sycl::write_only};
        oneapiext::single_task(CGH, KernelSingleTask, (int)N, 42, MemAcc);
      }).wait();
    }
  }
  for (size_t I = 0; I < N; ++I)
    Failed += Check(Memory, 42, I, "single_task");

  // 1D parallel_for
  {
    sycl::buffer<int, 1> MemBuf{Memory, sycl::range<1>{N}};
    {
      oneapiext::submit_with_event(Q, [&](sycl::handler &CGH) {
        sycl::accessor MemAcc{MemBuf, CGH, sycl::write_only};
        oneapiext::parallel_for(CGH, sycl::range<1>{N}, Kernel1D, 43, MemAcc);
      }).wait();
    }
  }
  for (size_t I = 0; I < N; ++I)
    Failed += Check(Memory, 43, I, "1D parallel_for");

  // 2D parallel_for
  {
    sycl::range<2> Range{8, N / 8};
    sycl::buffer<int, 2> MemBuf{Memory, Range};
    {
      oneapiext::submit_with_event(Q, [&](sycl::handler &CGH) {
        sycl::accessor MemAcc{MemBuf, CGH, sycl::write_only};
        oneapiext::parallel_for(CGH, Range, Kernel2D, 44, MemAcc);
      }).wait();
    }
  }
  for (size_t I = 0; I < N; ++I)
    Failed += Check(Memory, 44, I, "2D parallel_for");

  // 3D parallel_for
  {
    sycl::range<3> Range{8, 8, N / 64};
    sycl::buffer<int, 3> MemBuf{Memory, Range};
    {
      oneapiext::submit_with_event(Q, [&](sycl::handler &CGH) {
        sycl::accessor MemAcc{MemBuf, CGH, sycl::write_only};
        oneapiext::parallel_for(CGH, Range, Kernel3D, 45, MemAcc);
      }).wait();
    }
  }
  for (size_t I = 0; I < N; ++I)
    Failed += Check(Memory, 45, I, "3D parallel_for");

  // 1D nd_launch
  {
    sycl::buffer<int, 1> MemBuf{Memory, sycl::range<1>{N}};
    {
      oneapiext::submit_with_event(Q, [&](sycl::handler &CGH) {
        sycl::accessor MemAcc{MemBuf, CGH, sycl::write_only};
        oneapiext::nd_launch(
            CGH, sycl::nd_range<1>{sycl::range<1>{N}, sycl::range{8}}, Kernel1D,
            46, MemAcc);
      }).wait();
    }
  }
  for (size_t I = 0; I < N; ++I)
    Failed += Check(Memory, 46, I, "1D nd_launch");

  // 2D nd_launch
  {
    sycl::range<2> Range{8, N / 8};
    sycl::buffer<int, 2> MemBuf{Memory, Range};
    {
      oneapiext::submit_with_event(Q, [&](sycl::handler &CGH) {
        sycl::accessor MemAcc{MemBuf, CGH, sycl::write_only};
        oneapiext::nd_launch(CGH, sycl::nd_range<2>{Range, sycl::range{8, 8}},
                             Kernel2D, 47, MemAcc);
      }).wait();
    }
  }
  for (size_t I = 0; I < N; ++I)
    Failed += Check(Memory, 47, I, "2D nd_launch");

  // 3D nd_launch
  {
    sycl::range<3> Range{8, 8, N / 64};
    sycl::buffer<int, 3> MemBuf{Memory, Range};
    {
      oneapiext::submit_with_event(Q, [&](sycl::handler &CGH) {
        sycl::accessor MemAcc{MemBuf, CGH, sycl::write_only};
        oneapiext::nd_launch(CGH,
                             sycl::nd_range<3>{Range, sycl::range{8, 8, 8}},
                             Kernel3D, 48, MemAcc);
      }).wait();
    }
  }
  for (size_t I = 0; I < N; ++I)
    Failed += Check(Memory, 48, I, "3D nd_launch");

  // 1D parallel_for with launch config
  {
    sycl::buffer<int, 1> MemBuf{Memory, sycl::range<1>{N}};
    {
      oneapiext::submit_with_event(Q, [&](sycl::handler &CGH) {
        sycl::accessor MemAcc{MemBuf, CGH, sycl::write_only};
        oneapiext::parallel_for(
            CGH, oneapiext::launch_config<sycl::range<1>>{sycl::range<1>{N}},
            Kernel1D, 49, MemAcc);
      }).wait();
    }
  }
  for (size_t I = 0; I < N; ++I)
    Failed += Check(Memory, 49, I, "1D parallel_for with launch config");

  // 2D parallel_for with launch config
  {
    sycl::range<2> Range{8, N / 8};
    sycl::buffer<int, 2> MemBuf{Memory, Range};
    {
      oneapiext::submit_with_event(Q, [&](sycl::handler &CGH) {
        sycl::accessor MemAcc{MemBuf, CGH, sycl::write_only};
        oneapiext::parallel_for(CGH,
                                oneapiext::launch_config<sycl::range<2>>{Range},
                                Kernel2D, 50, MemAcc);
      }).wait();
    }
  }
  for (size_t I = 0; I < N; ++I)
    Failed += Check(Memory, 50, I, "2D parallel_for with launch config");

  // 3D parallel_for with launch config
  {
    sycl::range<3> Range{8, 8, N / 64};
    sycl::buffer<int, 3> MemBuf{Memory, Range};
    {
      oneapiext::submit_with_event(Q, [&](sycl::handler &CGH) {
        sycl::accessor MemAcc{MemBuf, CGH, sycl::write_only};
        oneapiext::parallel_for(CGH,
                                oneapiext::launch_config<sycl::range<3>>{Range},
                                Kernel3D, 51, MemAcc);
      }).wait();
    }
  }
  for (size_t I = 0; I < N; ++I)
    Failed += Check(Memory, 51, I, "3D parallel_for with launch config");

  // 1D nd_launch with launch config
  {
    sycl::buffer<int, 1> MemBuf{Memory, sycl::range<1>{N}};
    {
      oneapiext::submit_with_event(Q, [&](sycl::handler &CGH) {
        sycl::accessor MemAcc{MemBuf, CGH, sycl::write_only};
        oneapiext::nd_launch(
            CGH,
            oneapiext::launch_config<sycl::nd_range<1>>{
                sycl::nd_range<1>{sycl::range<1>{N}, sycl::range{8}}},
            Kernel1D, 52, MemAcc);
      }).wait();
    }
  }
  for (size_t I = 0; I < N; ++I)
    Failed += Check(Memory, 52, I, "1D nd_launch with launch config");

  // 2D nd_launch with launch config
  {
    sycl::range<2> Range{8, N / 8};
    sycl::buffer<int, 2> MemBuf{Memory, Range};
    {
      oneapiext::submit_with_event(Q, [&](sycl::handler &CGH) {
        sycl::accessor MemAcc{MemBuf, CGH, sycl::write_only};
        oneapiext::nd_launch(CGH,
                             oneapiext::launch_config<sycl::nd_range<2>>{
                                 sycl::nd_range<2>{Range, sycl::range{8, 8}}},
                             Kernel2D, 53, MemAcc);
      }).wait();
    }
  }
  for (size_t I = 0; I < N; ++I)
    Failed += Check(Memory, 53, I, "2D nd_launch with launch config");

  // 3D nd_launch with launch config
  {
    sycl::range<3> Range{8, 8, N / 64};
    sycl::buffer<int, 3> MemBuf{Memory, Range};
    {
      oneapiext::submit_with_event(Q, [&](sycl::handler &CGH) {
        sycl::accessor MemAcc{MemBuf, CGH, sycl::write_only};
        oneapiext::nd_launch(
            CGH,
            oneapiext::launch_config<sycl::nd_range<3>>{
                sycl::nd_range<3>{Range, sycl::range{8, 8, 8}}},
            Kernel3D, 54, MemAcc);
      }).wait();
    }
  }
  for (size_t I = 0; I < N; ++I)
    Failed += Check(Memory, 54, I, "3D nd_launch with launch config");

  return Failed;
}
