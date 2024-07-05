// REQUIRES: aspect-queue_profiling
// RUN: %{build} -o %t.out
//
// RUN: %{run} %t.out
//==------------------- event_profiling_info.cpp ---------------------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// Fails there.
// UNSUPPORTED: opencl && gpu && gpu-intel-pvc

#include <cassert>
#include <iostream>
#include <sycl/detail/core.hpp>
#include <sycl/properties/all_properties.hpp>

using namespace sycl;

bool verifyProfiling(event Event) {
  auto Submit =
      Event.get_profiling_info<sycl::info::event_profiling::command_submit>();
  auto Start =
      Event.get_profiling_info<sycl::info::event_profiling::command_start>();
  auto End =
      Event.get_profiling_info<sycl::info::event_profiling::command_end>();

  assert(Submit <= Start);
  assert(Start <= End);

  bool Pass = sycl::info::event_command_status::complete ==
              Event.get_info<sycl::info::event::command_execution_status>();

  return Pass;
}

// The test checks that get_profiling_info waits for command asccociated with
// event to complete execution.
int main() {
  device Dev;

  const size_t Size = 10000;
  int Data[Size] = {0};
  for (size_t I = 0; I < Size; ++I) {
    Data[I] = I;
  }
  int Values[Size] = {0};

  {
    buffer<int, 1> BufferFrom(Data, range<1>(Size));
    buffer<int, 1> BufferTo(Values, range<1>(Size));

    // buffer copy
    queue copyQueue{Dev, sycl::property::queue::enable_profiling()};
    event copyEvent = copyQueue.submit([&](sycl::handler &Cgh) {
      accessor<int, 1, access::mode::read, access::target::device> AccessorFrom(
          BufferFrom, Cgh, range<1>(Size));
      accessor<int, 1, access::mode::write, access::target::device> AccessorTo(
          BufferTo, Cgh, range<1>(Size));
      Cgh.copy(AccessorFrom, AccessorTo);
    });

    // kernel launch
    queue kernelQueue{Dev, sycl::property::queue::enable_profiling()};
    event kernelEvent = kernelQueue.submit([&](sycl::handler &CGH) {
      CGH.single_task<class EmptyKernel>([=]() {});
    });
    copyEvent.wait();
    kernelEvent.wait();

    // Profiling with HIP / CUDA backend may produce invalid results.
    // The below check on backend type should be removed once
    // https://github.com/intel/llvm/issues/10042 is resolved.
    if (kernelQueue.get_backend() != sycl::backend::ext_oneapi_cuda &&
        kernelQueue.get_backend() != sycl::backend::ext_oneapi_hip)
      assert(verifyProfiling(copyEvent) && verifyProfiling(kernelEvent));
  }

  for (size_t I = 0; I < Size; ++I) {
    assert(Data[I] == Values[I]);
  }

  return 0;
}
