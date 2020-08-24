// RUN: %clangxx -fsycl %s -o %t.out
// RUN: env SYCL_PI_TRACE=2 %CPU_RUN_PLACEHOLDER %t.out 2>&1 %CPU_CHECK_PLACEHOLDER
// RUN: env SYCL_PI_TRACE=2 %GPU_RUN_PLACEHOLDER %t.out 2>&1 %GPU_CHECK_PLACEHOLDER
// RUN: env SYCL_PI_TRACE=2 %ACC_RUN_PLACEHOLDER %t.out 2>&1 %ACC_CHECK_PLACEHOLDER

// UNSUPPORTED: cuda

#include <CL/sycl.hpp>
#include <CL/sycl/INTEL/fpga_device_selector.hpp>

int main() {
  sycl::context Context;
  sycl::queue Q1(Context, sycl::default_selector{});

  Q1.submit([&](sycl::handler &cgh) {
    cgh.single_task<class kernel1>([]() {});
  });
  Q1.submit([&](sycl::handler &cgh) {
    cgh.single_task<class kernel2>([]() {});
  });

  // call handler::barrier()
  Q1.submit([&](sycl::handler &cgh) {
    cgh.barrier();
  });

  Q1.submit([&](sycl::handler &cgh) {
    cgh.single_task<class kernel3>([]() {});
  });
  Q1.submit([&](sycl::handler &cgh) {
    cgh.single_task<class kernel4>([]() {});
  });

  // call queue::submit_barrier()
  Q1.submit_barrier();

  sycl::queue Q2(Context, sycl::default_selector{});
  sycl::queue Q3(Context, sycl::default_selector{});

  auto Event1 = Q1.submit([&](sycl::handler &cgh) {
    cgh.single_task<class kernel5>([]() {});
  });

  auto Event2 = Q2.submit([&](sycl::handler &cgh) {
    cgh.single_task<class kernel6>([]() {});
  });

  // call handler::barrier(const vector_class<event> &WaitList)
  Q3.submit([&](cl::sycl::handler &cgh) {
    cgh.barrier({Event1, Event2});
  });

  Q3.submit([&](sycl::handler &cgh) {
    cgh.single_task<class kernel7>([]() {});
  });

  auto Event3 = Q1.submit([&](sycl::handler &cgh) {
    cgh.single_task<class kernel8>([]() {});
  });

  auto Event4 = Q2.submit([&](sycl::handler &cgh) {
    cgh.single_task<class kernel9>([]() {});
  });

  // call queue::submit_barrier(const vector_class<event> &WaitList)
  Q3.submit_barrier({Event3, Event4});

  Q3.submit([&](sycl::handler &cgh) {
    cgh.single_task<class kernel10>([]() {});
  });

  return 0;
}

// CHECK:---> piEnqueueEventsWaitWithBarrier
// CHECK:---> piEnqueueEventsWaitWithBarrier
// CHECK:---> piEnqueueEventsWaitWithBarrier
// CHECK:---> piEnqueueEventsWaitWithBarrier
