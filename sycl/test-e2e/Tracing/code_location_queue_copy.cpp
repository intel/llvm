// REQUIRES: cpu
// UNSUPPORTED: windows
// RUN: %{build} -o %t.out
// RUN: %{run} sycl-trace --sycl %t.out | FileCheck %s

// Test tracing of the code location data for queue.copy in case of failure
// (exception generation)

#include <sycl/detail/core.hpp>
#include <sycl/usm.hpp>

int main() {
  sycl::queue Q;
  bool ExceptionCaught = false;
  unsigned char *HostAllocSrc = (unsigned char *)sycl::malloc_host(1, Q);
  unsigned char *HostAllocDst = NULL;
  try {
// CHECK: code_location_queue_copy.cpp:[[# @LINE + 1 ]] main
    Q.copy(HostAllocDst, HostAllocSrc, 1);
  } catch (sycl::exception &e) {
    std::ignore = e;
    ExceptionCaught = true;
  }
  Q.wait();
  sycl::free(HostAllocSrc, Q);

  return !ExceptionCaught;
}
