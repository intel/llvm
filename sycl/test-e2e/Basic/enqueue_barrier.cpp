// RUN: %{build} -o %t.out
// RUN: env SYCL_UR_TRACE=2 %{run} %t.out 2>&1 | FileCheck %s

#include <sycl/ext/intel/fpga_device_selector.hpp>
#include <sycl/detail/core.hpp>

int main() {
  sycl::context Context;
  sycl::queue Q1(Context, sycl::default_selector_v);

  Q1.submit(
      [&](sycl::handler &cgh) { cgh.single_task<class kernel1>([]() {}); });
  Q1.submit(
      [&](sycl::handler &cgh) { cgh.single_task<class kernel2>([]() {}); });

  // call handler::barrier()
  Q1.submit([&](sycl::handler &cgh) { cgh.ext_oneapi_barrier(); });

  Q1.submit(
      [&](sycl::handler &cgh) { cgh.single_task<class kernel3>([]() {}); });
  Q1.submit(
      [&](sycl::handler &cgh) { cgh.single_task<class kernel4>([]() {}); });

  // call queue::ext_oneapi_submit_barrier()
  Q1.ext_oneapi_submit_barrier();

  sycl::queue Q2(Context, sycl::default_selector_v);
  sycl::queue Q3(Context, sycl::default_selector_v);

  auto Event1 = Q1.submit(
      [&](sycl::handler &cgh) { cgh.single_task<class kernel5>([]() {}); });

  auto Event2 = Q2.submit(
      [&](sycl::handler &cgh) { cgh.single_task<class kernel6>([]() {}); });

  // call handler::barrier(const std::vector<event> &WaitList)
  Q3.submit([&](sycl::handler &cgh) {
    cgh.ext_oneapi_barrier({Event1, Event2});
  });

  Q3.submit(
      [&](sycl::handler &cgh) { cgh.single_task<class kernel7>([]() {}); });

  auto Event3 = Q1.submit(
      [&](sycl::handler &cgh) { cgh.single_task<class kernel8>([]() {}); });

  auto Event4 = Q2.submit(
      [&](sycl::handler &cgh) { cgh.single_task<class kernel9>([]() {}); });

  // call queue::ext_oneapi_submit_barrier(const std::vector<event> &WaitList)
  Q3.ext_oneapi_submit_barrier({Event3, Event4});

  Q3.submit(
      [&](sycl::handler &cgh) { cgh.single_task<class kernel10>([]() {}); });

  return 0;
}

// CHECK: <--- urEnqueueEventsWaitWithBarrier
// CHECK: <--- urEnqueueEventsWaitWithBarrier
// CHECK: <--- urEnqueueEventsWaitWithBarrier
// CHECK: <--- urEnqueueEventsWaitWithBarrier
