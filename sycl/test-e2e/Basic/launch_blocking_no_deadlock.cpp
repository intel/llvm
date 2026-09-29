// The cases SYCL_LAUNCH_BLOCKING does not make synchronous, because a drain
// there would wait on work that only completes once the submitting thread
// continues. This test checks for deadlock to make sure SYCL_LAUNCH_BLOCKING
// does not introduce deadlock with SYCL RT's internal synchronization.
//
// REQUIRES: aspect-usm_shared_allocations
//
// RUN: %{build} -o %t.out
// RUN: env SYCL_LAUNCH_BLOCKING=1 %{run} %t.out

#include <cassert>
#include <future>
#include <sycl/detail/core.hpp>
#include <sycl/properties/queue_properties.hpp>
#include <sycl/stream.hpp>
#include <sycl/usm.hpp>
#include <thread>

// A stream makes submit_impl recurse to submit its flush host task.
static void runStreamFlushHostTask(sycl::queue &Q) {
  Q.submit([&](sycl::handler &CGH) {
    sycl::stream OS{1024, 256, CGH};
    CGH.single_task([=]() { OS << 1 << sycl::endl; });
  });
  Q.wait();
}

// A kernel depending on a host task is enqueued by the host task's completion
// callback, on another thread - blocking applies where the command is enqueued
// to UR, not where it is submitted, so the event is not complete on return.
static void runKernelAfterHostTask(sycl::queue &Q) {
  int *Out = sycl::malloc_shared<int>(1, Q);
  *Out = 0;
  sycl::event HostEvent =
      Q.submit([&](sycl::handler &CGH) { CGH.host_task([]() {}); });
  Q.submit([&](sycl::handler &CGH) {
    CGH.depends_on(HostEvent);
    CGH.single_task([=]() { Out[0] = 1; });
  });
  Q.wait();
  assert(*Out == 1);
  sycl::free(Out, Q);
}

// A host task that only finishes once the submitting thread continues.
static void runGatedHostTask(sycl::queue &Q) {
  std::promise<void> Gate;
  std::future<void> Gated = Gate.get_future();
  Q.submit(
      [&](sycl::handler &CGH) { CGH.host_task([&Gated]() { Gated.wait(); }); });
  Gate.set_value();
  Q.wait();
}

// Submitting to a queue from inside a host task running on it.
static void runSubmitFromHostTask(sycl::queue &Q) {
  int *Out = sycl::malloc_shared<int>(1, Q);
  *Out = 0;
  Q.submit([&](sycl::handler &CGH) {
    CGH.host_task([&]() { Q.single_task([=]() { Out[0] = 1; }); });
  });
  Q.wait();
  Q.wait(); // The inner submission was made while the first wait ran.
  assert(*Out == 1);
  sycl::free(Out, Q);
}

// A thread blocked in a host task must not hold up another thread.
static void runSharedQueue(sycl::queue &Q) {
  std::promise<void> Gate, Running;
  std::future<void> Gated = Gate.get_future();
  std::future<void> IsRunning = Running.get_future();
  std::thread Blocked{[&]() {
    Q.submit([&](sycl::handler &CGH) {
      CGH.host_task([&]() {
        Running.set_value();
        Gated.wait();
      });
    });
  }};

  IsRunning.wait(); // Submit only once the host task is blocking.
  int *Out = sycl::malloc_device<int>(1, Q);
  Q.single_task([=]() { *Out = 1; });

  Gate.set_value();
  Blocked.join();
  Q.wait();
  sycl::free(Out, Q);
}

int main() {
  // Both queue kinds: they take different submission paths.
  sycl::queue InOrder{sycl::property::queue::in_order{}};
  sycl::queue OutOfOrder;
  for (sycl::queue *Q : {&InOrder, &OutOfOrder}) {
    runStreamFlushHostTask(*Q);
    runKernelAfterHostTask(*Q);
  }

  runGatedHostTask(InOrder);
  runSubmitFromHostTask(InOrder);
  runSharedQueue(OutOfOrder);
  return 0;
}
