// UNSUPPORTED: windows || hip_amd
// RUN: %{build} -o %t.out
// RUN: not env SYCL_TRACE_TERMINATE_ON_WARNING=1 %{run} sycl-trace --verify %t.out | FileCheck %s

// Test parameter analysis of USM usage

#include <sycl/detail/core.hpp>
#include <sycl/usm.hpp>

int main() {
  sycl::queue Q;
  unsigned int *AllocSrc = (unsigned int *)sycl::malloc_device(sizeof(int), Q);
  sycl::free(AllocSrc, Q);

  try {
    // CHECK: [USM] Function uses unknown USM pointer (could be already released or not allocated as USM) as kernel parameter with index = 0.
    // CHECK: | kernel location: function main at {{.*}}queue_single_task_released_pointer.cpp:[[# @LINE + 1 ]]
    Q.single_task([=]() { *AllocSrc = 13; });
    Q.wait();
  } catch (...) {
  }
  return 0;
}
