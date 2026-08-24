// Verifies that SYCL_LAUNCH_BLOCKING makes kernel and host_task submissions
// synchronous: the queue must be idle by the time the submission call returns.
//
// RUN: %{build} -o %t.out
// RUN: %{run} %t.out
// RUN: env SYCL_LAUNCH_BLOCKING=1 %{run} %t.out
// RUN: env SYCL_LAUNCH_BLOCKING=2 %{run} %t.out

#include <cassert>
#include <cstdlib>
#include <iostream>
#include <sycl/detail/core.hpp>
#include <sycl/ext/oneapi/experimental/enqueue_functions.hpp>
#include <sycl/properties/queue_properties.hpp>
#include <sycl/reduction.hpp>
#include <sycl/usm.hpp>

namespace exp_ext = sycl::ext::oneapi::experimental;

// Sized so that the work is still in flight right after an asynchronous
// submission returns, but does not noticeably slow the test down.
constexpr int Iterations = 20000000;
constexpr size_t N = 1024;

static const bool Blocking = []() {
  const char *Val = std::getenv("SYCL_LAUNCH_BLOCKING");
  return Val && std::atoi(Val) != 0;
}();

// With blocking enabled the queue must be empty the moment the submission call
// returns. Without it, early completion is legal, so only run the check for the
// blocking configuration.
static void checkIdle(sycl::queue &Q, const char *What) {
  const bool Idle = Q.ext_oneapi_empty();
  if (Blocking && !Idle)
    std::cerr << "queue not idle after " << What << std::endl;
  assert((!Blocking || Idle) && "submission was not synchronous");
  Q.wait();
}

// Spins long enough on the device to be observable, writing a result the
// compiler cannot fold away.
static void spin(int *Out, size_t Idx) {
  int Acc = 0;
  for (int I = 0; I < Iterations / static_cast<int>(N); ++I)
    Acc += I % 7;
  Out[Idx] = Acc;
}

static void runOnQueue(sycl::queue &Q, const char *Order) {
  int *Out = sycl::malloc_device<int>(N, Q);
  int *Sum = sycl::malloc_device<int>(1, Q);
  Q.memset(Sum, 0, sizeof(int)).wait();

  // handler parallel_for: queue_impl::submit_impl.
  Q.submit([&](sycl::handler &CGH) {
    CGH.parallel_for(sycl::range<1>{N},
                     [=](sycl::id<1> Idx) { spin(Out, Idx); });
  });
  checkIdle(Q, "submit + parallel_for");

  // Queue kernel shortcuts: queue_impl::submit_kernel_direct_impl.
  Q.parallel_for(sycl::range<1>{N}, [=](sycl::id<1> Idx) { spin(Out, Idx); });
  checkIdle(Q, "parallel_for shortcut");

  Q.single_task([=]() { spin(Out, 0); });
  checkIdle(Q, "single_task shortcut");

  // Free function enqueue API, nd_range form.
  exp_ext::nd_launch(
      Q, sycl::nd_range<1>{sycl::range<1>{N}, sycl::range<1>{64}},
      [=](sycl::nd_item<1> It) { spin(Out, It.get_global_id(0)); });
  checkIdle(Q, "nd_launch");

  // A reduction submits runtime-internal kernels around the user kernel; all of
  // them must be covered by the outer submission's wait.
  Q.submit([&](sycl::handler &CGH) {
    CGH.parallel_for(sycl::range<1>{N}, sycl::reduction(Sum, sycl::plus<int>()),
                     [=](sycl::id<1>, auto &Reducer) { Reducer += 1; });
  });
  checkIdle(Q, "reduction");

  int HostSum = 0;
  Q.memcpy(&HostSum, Sum, sizeof(int)).wait();
  assert(HostSum == static_cast<int>(N) && "reduction produced a wrong result");

  // host_task takes its own path through submit_impl.
  Q.submit([&](sycl::handler &CGH) {
    CGH.host_task([]() {
      volatile int Acc = 0;
      for (int I = 0; I < Iterations; ++I)
        Acc += I % 7;
    });
  });
  checkIdle(Q, "host_task");

  // A kernel depending on a host task cannot bypass the scheduler, so this
  // exercises the scheduler-based branch of submit_impl.
  sycl::event HostEvent =
      Q.submit([&](sycl::handler &CGH) { CGH.host_task([]() {}); });
  Q.submit([&](sycl::handler &CGH) {
    CGH.depends_on(HostEvent);
    CGH.parallel_for(sycl::range<1>{N},
                     [=](sycl::id<1> Idx) { spin(Out, Idx); });
  });
  checkIdle(Q, "kernel after host_task");

  sycl::free(Sum, Q);
  sycl::free(Out, Q);
  std::cout << Order << " queue: OK" << std::endl;
}

int main() {
  sycl::queue InOrder{sycl::property::queue::in_order{}};
  runOnQueue(InOrder, "in-order");

  sycl::queue OutOfOrder;
  runOnQueue(OutOfOrder, "out-of-order");

  return 0;
}
