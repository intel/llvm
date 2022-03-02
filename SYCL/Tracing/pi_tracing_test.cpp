// RUN: %clangxx -fsycl -fsycl-targets=%sycl_triple %s -o %t.out
// RUN: env SYCL_PI_TRACE=-1 %CPU_RUN_PLACEHOLDER %t.out %CPU_CHECK_PLACEHOLDER
// RUN: env SYCL_PI_TRACE=-1 %GPU_RUN_PLACEHOLDER %t.out %GPU_CHECK_PLACEHOLDER
// RUN: env SYCL_PI_TRACE=-1 %ACC_RUN_PLACEHOLDER %t.out %ACC_CHECK_PLACEHOLDER
//
// XFAIL: hip_nvidia

// Test tracing of the Plugin Interface

// CHECK: ---> piPlatformGetInfo(
// CHECK:  pi_platform : {{0[xX]?[0-9a-fA-F]*}}
// CHECK:  <char * > : {{0[xX]?[0-9a-fA-F]*}}
// CHECK:  <nullptr>
// CHECK: ---> piMemBufferCreate(
// CHECK-NEXT:  <unknown> : {{0[xX]?[0-9a-fA-F]*}}
// CHECK-NEXT:  <unknown> : 1
// CHECK-NEXT:  <unknown> : 40
// CHECK-NEXT:  <unknown> : 0
// CHECK-NEXT:  <unknown> : {{0[xX]?[0-9a-fA-F]*}}
// CHECK:  {{<nullptr>|0}}
// CHECK-NEXT: ) ---> 	pi_result : PI_SUCCESS
// CHECK-NEXT: [out]void * : {{0+}}
// CHECK-NEXT: [out]pi_mem * : {{0[xX]?[0-9a-fA-F]*}}[ {{0[xX]?[0-9a-fA-F]*}}
// CHECK: ---> piKernelCreate(
// CHECK:  <const char *>: {{.*}}
// CHECK: ---> piEnqueueKernelLaunch(
// CHECK-NEXT:  <unknown> : {{0[xX]?[0-9a-fA-F]*}}
// CHECK-NEXT:  <unknown> : {{0[xX]?[0-9a-fA-F]*}}
// CHECK-NEXT:  <unknown> : 1
// CHECK-NEXT:  <unknown> : {{0[xX]?[0-9a-fA-F]*}}
// CHECK-NEXT:  <unknown> : {{0[xX]?[0-9a-fA-F]*}}
// CHECK-NEXT:  <unknown> : 0
// CHECK-NEXT:  <unknown> : 0
// CHECK-NEXT:  pi_event * : {{0+}}[ nullptr ]
// CHECK-NEXT:  pi_event * : {{0[xX]?[0-9a-fA-F]*}}[ {{0+}} ... ]
// CHECK-NEXT: ) ---> 	pi_result : PI_SUCCESS
// CHECK-NEXT: [out]pi_event * : {{0+}}[ nullptr ]
// CHECK-NEXT: [out]pi_event * : {{0[xX]?[0-9a-fA-F]*}}
// CHECK-SAME: [ {{0[xX]?[0-9a-fA-F]*}} ... ]
//
// CHECK: ---> piEventsWait(
// CHECK-NEXT:  <unknown> : 1
// CHECK-NEXT:  {{(const |\[out\])?}}pi_event * : {{0[xX]?[0-9a-fA-F]*}}
// CHECK-SAME:  [ {{0[xX]?[0-9a-fA-F]*}} ... ]
// CHECK-NEXT: ) ---> 	pi_result : PI_SUCCESS

#include <CL/sycl.hpp>
int main() {
  cl::sycl::queue Queue;
  cl::sycl::buffer<int, 1> Buf(10);
  cl::sycl::event E = Queue.submit([&](cl::sycl::handler &cgh) {
    auto Acc = Buf.template get_access<cl::sycl::access::mode::read_write>(cgh);

    cgh.parallel_for<class CheckTraces>(
        10, [=](cl::sycl::id<1> ID) { Acc[ID] = 5; });
  });
  E.wait();
  return 0;
}
