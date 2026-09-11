// SYCL_LAUNCH_BLOCKING makes the device work of a submission synchronous, so
// the event of a submission is already complete when the submission returns.
//
// The blocking is done by a UR layer, which cannot see a host task. That is
// what keeps a host task whose completion depends on host code running after
// the submission - and a submission from inside a host task, and a queue shared
// with a thread that is blocked in one - from deadlocking. Every case below
// hangs if that stops holding.
//
// REQUIRES: aspect-usm_shared_allocations
//
// RUN: %{build} -o %t.out
// RUN: env SYCL_LAUNCH_BLOCKING=1 %{run} %t.out

#include <cassert>
#include <future>
#include <sycl/detail/core.hpp>
#include <sycl/properties/queue_properties.hpp>
#include <sycl/usm.hpp>
#include <thread>

constexpr size_t N = 1024;

static bool isComplete(sycl::event E) {
  return E.get_info<sycl::info::event::command_execution_status>() ==
         sycl::info::event_command_status::complete;
}

// The device work of a submission has completed by the time it returns, so its
// event is complete and its result is readable without waiting.
static void runDeviceCommands(sycl::queue &Q) {
  int *Out = sycl::malloc_shared<int>(N, Q);

  sycl::event Fill = Q.fill(Out, 1, N);
  assert(isComplete(Fill) && "memory operation did not block");
  assert(Out[0] == 1 && Out[N - 1] == 1);

  sycl::event Kernel = Q.submit([&](sycl::handler &CGH) {
    CGH.parallel_for(sycl::range<1>{N},
                     [=](sycl::id<1> Idx) { Out[Idx] += 1; });
  });
  assert(isComplete(Kernel) && "handler submission did not block");
  assert(Out[0] == 2 && Out[N - 1] == 2);

  // Kernel shortcut, i.e. the scheduler-bypass path.
  sycl::event Shortcut = Q.parallel_for(
      sycl::range<1>{N}, [=](sycl::id<1> Idx) { Out[Idx] += 1; });
  assert(isComplete(Shortcut) && "kernel shortcut did not block");
  assert(Out[0] == 3 && Out[N - 1] == 3);

  // A barrier submits no work of its own and is not made synchronous, but it
  // must still order the commands around it.
  Q.ext_oneapi_submit_barrier();
  Q.parallel_for(sycl::range<1>{N}, [=](sycl::id<1> Idx) { Out[Idx] += 1; });
  Q.wait();
  assert(Out[0] == 4 && Out[N - 1] == 4);

  sycl::free(Out, Q);
}

// A host task that can only finish once the submitting thread continues.
static void runGatedHostTask(sycl::queue &Q) {
  std::promise<void> Gate;
  std::future<void> Gated = Gate.get_future();

  Q.submit(
      [&](sycl::handler &CGH) { CGH.host_task([&Gated]() { Gated.wait(); }); });

  Gate.set_value();
  Q.wait();
}

// Submitting to a queue from inside a host task running on that queue.
static void runSubmitFromHostTask(sycl::queue &Q) {
  int *Out = sycl::malloc_shared<int>(1, Q);
  *Out = 0;

  Q.submit([&](sycl::handler &CGH) {
    CGH.host_task([&]() { Q.single_task([=]() { Out[0] = 1; }); });
  });

  Q.wait();
  // The inner submission was made while the wait above was already running.
  Q.wait();
  assert(*Out == 1 && "the submission made inside the host task did not run");

  sycl::free(Out, Q);
}

// One thread blocked in a host task must not hold up another thread's
// submissions to the same queue.
static void runSharedQueue(sycl::queue &Q) {
  std::promise<void> Gate;
  std::future<void> Gated = Gate.get_future();

  std::thread Blocked{[&]() {
    Q.submit([&](sycl::handler &CGH) {
      CGH.host_task([&Gated]() { Gated.wait(); });
    });
  }};

  int *Out = sycl::malloc_device<int>(1, Q);
  Q.single_task([=]() { *Out = 1; });

  Gate.set_value();
  Blocked.join();
  Q.wait();
  sycl::free(Out, Q);
}

int main() {
  sycl::queue InOrder{sycl::property::queue::in_order{}};
  runDeviceCommands(InOrder);

  sycl::queue OutOfOrder;
  runDeviceCommands(OutOfOrder);

  runGatedHostTask(InOrder);
  runSubmitFromHostTask(InOrder);
  runSharedQueue(OutOfOrder);

  return 0;
}
