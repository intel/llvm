// REQUIRES: cpu
// UNSUPPORTED: windows
// RUN: %{build} -o %t.out
// RUN: %{run} sycl-trace --sycl --print-format=verbose %t.out | FileCheck %s

// Test tracing of the code location data for queue.submit in case of failure
// (exception generation)
// First queue creation (id = 0) is queue created on line 17.

#include <sycl/detail/core.hpp>
#include <sycl/usm.hpp>

int main() {
  bool ExceptionCaught = false;
  {
    sycl::queue Q;
    unsigned char *HostAllocSrc = (unsigned char *)sycl::malloc_host(1, Q);
    unsigned char *HostAllocDst = NULL;
    // CHECK: [SYCL] Queue create:
    // CHECK-DAG:        queue_handle : {{.*}}
    // CHECK-DAG:        queue_id : 1
    // CHECK-DAG:        is_inorder : false
    // CHECK-DAG:        sycl_device : {{.*}}
    // CHECK-DAG:        sycl_device_name : {{.*}}
    // CHECK-DAG:        sycl_context : {{.*}}
    // CHECK: [SYCL] Runtime reports:
    // CHECK-NEXT: what:  NULL pointer argument in memory copy operation. -30 (PI_ERROR_INVALID_VALUE)
    // CHECK-NEXT: where:{{.*}}code_location_queue_submit.cpp:[[# @LINE + 2 ]] main
    try {
      Q.submit(
          [&](sycl::handler &cgh) { cgh.copy(HostAllocDst, HostAllocSrc, 1); });
    } catch (sycl::exception &e) {
      std::ignore = e;
      ExceptionCaught = true;
    }
    Q.wait();
    sycl::free(HostAllocSrc, Q);
  }
  // CHECK-NEXT: [SYCL] Queue destroy:
  // CHECK-DAG:        queue_id : 1
  return !ExceptionCaught;
}
