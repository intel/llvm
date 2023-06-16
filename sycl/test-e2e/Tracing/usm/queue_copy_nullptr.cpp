// UNSUPPORTED: windows
// RUN: %{build} -o %t.out
// RUN: %{run} sycl-trace --plugin --verify %t.out | FileCheck %s

// Test parameter analysis of USM function

#include <sycl/sycl.hpp>

int main() {
  sycl::queue Q;
  unsigned int *AllocSrc = (unsigned int *)sycl::malloc_device(sizeof(int), Q);
  unsigned int *AllocDst = (unsigned int *)sycl::malloc_device(sizeof(int), Q);
  sycl::free(AllocDst, Q);

  try {
    // CHECK: [USM] Function uses unknown USM pointer (could be already released or not allocated as USM) as destination memory block
    // CHECK: | memcpy location: function main at {{.*}}/queue_copy_nullptr.cpp:18
    Q.copy(AllocDst, AllocSrc, 1);
  } catch (...) {
  }
  Q.wait();
  sycl::free(AllocSrc, Q);
  return 0;
}
