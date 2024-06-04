// RUN: %{build} -o %t.out
// RUN: env SYCL_PI_TRACE=2 %{run} %t.out 2>&1 | FileCheck %s

// Tests the enqueue free function barriers.

#include <sycl/detail/core.hpp>
#include <sycl/ext/oneapi/experimental/enqueue_functions.hpp>

namespace oneapiext = sycl::ext::oneapi::experimental;

int main() {
  sycl::context Context;
  sycl::queue Q1(Context, sycl::default_selector_v);

  oneapiext::single_task(Q1, []() {});
  oneapiext::single_task(Q1, []() {});

  oneapiext::barrier(Q1);

  oneapiext::single_task(Q1, []() {});
  oneapiext::single_task(Q1, []() {});

  oneapiext::barrier(Q1);

  sycl::queue Q2(Context, sycl::default_selector_v);
  sycl::queue Q3(Context, sycl::default_selector_v);

  sycl::event Event1 = oneapiext::submit_with_event(
      Q1, [&](sycl::handler &CGH) { oneapiext::single_task(CGH, []() {}); });

  sycl::event Event2 = oneapiext::submit_with_event(
      Q2, [&](sycl::handler &CGH) { oneapiext::single_task(CGH, []() {}); });

  oneapiext::partial_barrier(Q3, {Event1, Event2});

  oneapiext::single_task(Q3, []() {});

  sycl::event Event3 = oneapiext::submit_with_event(
      Q1, [&](sycl::handler &CGH) { oneapiext::single_task(CGH, []() {}); });

  sycl::event Event4 = oneapiext::submit_with_event(
      Q2, [&](sycl::handler &CGH) { oneapiext::single_task(CGH, []() {}); });

  oneapiext::partial_barrier(Q3, {Event3, Event4});

  oneapiext::single_task(Q3, []() {});

  Q1.wait();

  return 0;
}

// CHECK-COUNT-4:---> piEnqueueEventsWaitWithBarrier
// CHECK-NOT:---> piEnqueueEventsWaitWithBarrier
