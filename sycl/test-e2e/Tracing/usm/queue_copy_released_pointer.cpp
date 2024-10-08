// UNSUPPORTED: windows || hip_amd
// RUN: %{build} -o %t.out
// RUN: not env SYCL_TRACE_TERMINATE_ON_WARNING=1 %{run} sycl-trace --verify %t.out | FileCheck %s

// Test parameter analysis of USM function

#include <sycl/detail/core.hpp>
#include <sycl/usm.hpp>

int main() {
  sycl::queue Q;
  unsigned int *AllocSrc = (unsigned int *)sycl::malloc_device(sizeof(int), Q);
  unsigned int *AllocDst = (unsigned int *)sycl::malloc_device(sizeof(int), Q);

  sycl::free(AllocDst, Q);

  try {
    // CHECK: [USM] Function uses unknown USM pointer (could be already released or not allocated as USM) as destination memory block
    // CHECK: | memcpy location: function main at {{.*}}queue_copy_released_pointer.cpp:[[# @LINE + 1 ]]
    Q.copy(AllocSrc, AllocDst, 1);
    Q.wait();
  } catch (...) {
  }
  sycl::free(AllocSrc, Q);
  return 0;
}
