// UNSUPPORTED: windows
// RUN: %clangxx -fsycl -fsycl-targets=%sycl_triple %s -o %t.out
// RUN: %CPU_RUN_PLACEHOLDER sycl-trace --sycl %t.out %CPU_CHECK_PLACEHOLDER

// Test tracing of the code location data for queue.submit in case of failure
// (exception generation)
//
// CHECK: code_location_queue_submit.cpp:18 main

#include <sycl/sycl.hpp>

int main() {
  sycl::queue Q;
  bool ExceptionCaught = false;
  unsigned char *HostAllocSrc = (unsigned char *)sycl::malloc_host(1, Q);
  unsigned char *HostAllocDst = NULL;
  try {
    Q.submit(
        [&](sycl::handler &cgh) { cgh.copy(HostAllocDst, HostAllocSrc, 1); });
  } catch (sycl::exception &e) {
    std::ignore = e;
    ExceptionCaught = true;
  }
  Q.wait();
  sycl::free(HostAllocSrc, Q);

  return !ExceptionCaught;
}
