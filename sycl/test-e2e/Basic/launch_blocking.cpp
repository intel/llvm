// Verifies that SYCL_LAUNCH_BLOCKING makes kernel and host_task submissions
// synchronous: the result of a submission is already visible to the host by the
// time the submission call returns, without any wait.
//
// REQUIRES: aspect-usm_shared_allocations
//
// RUN: %{build} -o %t.out
// RUN: %{run} %t.out
// RUN: env SYCL_LAUNCH_BLOCKING=1 %{run} %t.out blocking
// RUN: env SYCL_LAUNCH_BLOCKING=2 %{run} %t.out blocking

#include <cassert>
#include <iostream>
#include <sycl/detail/core.hpp>
#include <sycl/ext/oneapi/experimental/enqueue_functions.hpp>
#include <sycl/properties/queue_properties.hpp>
#include <sycl/reduction.hpp>
#include <sycl/usm.hpp>

namespace exp_ext = sycl::ext::oneapi::experimental;

constexpr size_t N = 1024;

// Iteration count of the spin loop below. Long enough that an asynchronous
// submission is still in flight when it returns, short enough not to slow the
// test down noticeably on a CPU device.
constexpr int SpinCount = 20000;

// Set by the RUN lines that enable blocking mode. Without blocking a submission
// may legitimately have completed already, so the checks only apply when this
// is set.
static bool Blocking = false;

// The spin loop, also run on the host to get the expected result. The loop
// count is read from memory so that the device-side loop cannot be folded away.
static int spin(const int *Limit) {
  int Acc = 0;
  for (int I = 0; I < *Limit; ++I)
    Acc += I % 7;
  return Acc;
}

// Spins, then writes a result the caller can attribute to this submission.
static void spinAndTag(int *Out, size_t Idx, const int *Limit, int Tag) {
  Out[Idx] = spin(Limit) + Tag;
}

static void check(sycl::queue &Q, const int *Out, const int *Limit, int Tag,
                  const char *What) {
  if (Blocking) {
    const int Expected = spin(Limit) + Tag;
    if (Out[0] != Expected)
      std::cerr << "result of " << What << " is not visible yet" << std::endl;
    assert(Out[0] == Expected && "submission was not synchronous");
  }
  Q.wait();
}

static void runOnQueue(sycl::queue &Q, const char *Order) {
  int *Out = sycl::malloc_shared<int>(N, Q);
  int *Sum = sycl::malloc_shared<int>(1, Q);
  int *Limit = sycl::malloc_shared<int>(1, Q);
  *Limit = SpinCount;
  *Sum = 0;
  int Tag = 0;

  // handler parallel_for: queue_impl::submit_impl.
  ++Tag;
  Q.submit([&](sycl::handler &CGH) {
    CGH.parallel_for(sycl::range<1>{N}, [=](sycl::id<1> Idx) {
      spinAndTag(Out, Idx, Limit, Tag);
    });
  });
  check(Q, Out, Limit, Tag, "submit + parallel_for");

  // Queue kernel shortcuts: queue_impl::submit_kernel_direct_impl.
  ++Tag;
  Q.parallel_for(sycl::range<1>{N},
                 [=](sycl::id<1> Idx) { spinAndTag(Out, Idx, Limit, Tag); });
  check(Q, Out, Limit, Tag, "parallel_for shortcut");

  ++Tag;
  Q.single_task([=]() { spinAndTag(Out, 0, Limit, Tag); });
  check(Q, Out, Limit, Tag, "single_task shortcut");

  // Free function enqueue API, nd_range form. It does not return an event,
  // which takes the discard-event exit of the fast path.
  ++Tag;
  exp_ext::nd_launch(Q,
                     sycl::nd_range<1>{sycl::range<1>{N}, sycl::range<1>{64}},
                     [=](sycl::nd_item<1> It) {
                       spinAndTag(Out, It.get_global_id(0), Limit, Tag);
                     });
  check(Q, Out, Limit, Tag, "nd_launch");

  // A reduction submits runtime-internal kernels around the user kernel; all of
  // them must be covered by the outer submission's wait.
  ++Tag;
  Q.submit([&](sycl::handler &CGH) {
    CGH.parallel_for(sycl::range<1>{N}, sycl::reduction(Sum, sycl::plus<int>()),
                     [=](sycl::id<1> Idx, auto &Reducer) {
                       spinAndTag(Out, Idx, Limit, Tag);
                       Reducer += 1;
                     });
  });
  check(Q, Out, Limit, Tag, "reduction");
  assert(*Sum == static_cast<int>(N) && "reduction produced a wrong result");

  // host_task takes its own path through submit_impl.
  bool HostTaskDone = false;
  Q.submit([&](sycl::handler &CGH) {
    CGH.host_task([&HostTaskDone]() {
      volatile int Acc = 0;
      for (int I = 0; I < SpinCount * 100; ++I)
        Acc += I % 7;
      HostTaskDone = true;
    });
  });
  if (Blocking && !HostTaskDone)
    std::cerr << "host_task has not run yet" << std::endl;
  assert((!Blocking || HostTaskDone) && "submission was not synchronous");
  Q.wait();

  // A kernel depending on a host task cannot bypass the scheduler, so this
  // exercises the scheduler-based branch of submit_impl.
  ++Tag;
  sycl::event HostEvent =
      Q.submit([&](sycl::handler &CGH) { CGH.host_task([]() {}); });
  Q.submit([&](sycl::handler &CGH) {
    CGH.depends_on(HostEvent);
    CGH.parallel_for(sycl::range<1>{N}, [=](sycl::id<1> Idx) {
      spinAndTag(Out, Idx, Limit, Tag);
    });
  });
  check(Q, Out, Limit, Tag, "kernel after host_task");

  sycl::free(Limit, Q);
  sycl::free(Sum, Q);
  sycl::free(Out, Q);
  std::cout << Order << " queue: OK" << std::endl;
}

int main(int argc, char *argv[]) {
  Blocking = argc > 1;

  sycl::queue InOrder{sycl::property::queue::in_order{}};
  runOnQueue(InOrder, "in-order");

  sycl::queue OutOfOrder;
  runOnQueue(OutOfOrder, "out-of-order");

  return 0;
}
