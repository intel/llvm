// Barriers and markers are deliberately excluded from SYCL_LAUNCH_BLOCKING:
// they run no user work, so blocking on one adds no debugging value.
//
// This test checks that the whole barrier family still completes and still
// orders work correctly in blocking mode, on both queue kinds and through the
// queue, handler and free-function spellings. It is a liveness and correctness
// test: it does not assert that barriers stay asynchronous, since a barrier may
// legitimately complete immediately, and it does not prove the exclusion is
// required - see the comment on queue_impl::waitIfLaunchBlocking for that
// reasoning.
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
#include <sycl/ext/oneapi/experimental/reusable_events.hpp>
#include <sycl/properties/queue_properties.hpp>
#include <sycl/usm.hpp>
#include <vector>

namespace exp_ext = sycl::ext::oneapi::experimental;

constexpr size_t N = 1024;

static void runOnQueue(sycl::queue &Q, const char *Order) {
  int *Out = sycl::malloc_device<int>(N, Q);
  Q.fill(Out, 0, N).wait();

  auto bump = [&]() {
    return Q.parallel_for(sycl::range<1>{N},
                          [=](sycl::id<1> Idx) { Out[Idx] += 1; });
  };

  // Queue-level barrier with no dependencies.
  bump();
  Q.ext_oneapi_submit_barrier();

  // Queue-level barrier with an explicit wait list.
  sycl::event E = bump();
  Q.ext_oneapi_submit_barrier({E});

  // handler barrier inside a command group. This is the path that reaches
  // submit_impl with CGType::Barrier, i.e. the excluded type.
  bump();
  Q.submit([&](sycl::handler &CGH) { CGH.ext_oneapi_barrier(); });

  // handler barrier with a wait list: CGType::BarrierWaitlist.
  sycl::event E2 = bump();
  Q.submit([&](sycl::handler &CGH) { CGH.ext_oneapi_barrier({E2}); });

  // Free function forms.
  bump();
  exp_ext::barrier(Q);

  sycl::event E3 = bump();
  exp_ext::partial_barrier(Q, {E3});

  Q.wait();

  std::vector<int> Host(N, -1);
  Q.memcpy(Host.data(), Out, N * sizeof(int)).wait();
  for (int V : Host)
    assert(V == 6 && "barrier interfered with kernel results");

  sycl::free(Out, Q);
  std::cout << Order << " queue: OK" << std::endl;
}

// Reusable events cross two queues through enqueue_signal_event and
// enqueue_wait_event, which reach the barrier fast path rather than
// submit_impl. Blocking mode must not disturb the ordering they establish.
static void runReusableEventCase() {
  sycl::device Dev;
  sycl::context Ctx = Dev.get_platform().khr_get_default_context();
  sycl::queue Q1{Ctx, Dev, sycl::property::queue::in_order{}};
  sycl::queue Q2{Ctx, Dev, sycl::property::queue::in_order{}};

  int *Out = sycl::malloc_device<int>(N, Q1);
  Q1.fill(Out, 0, N).wait();

  sycl::event Reusable = exp_ext::make_event(Ctx);

  Q1.parallel_for(sycl::range<1>{N}, [=](sycl::id<1> Idx) { Out[Idx] += 1; });
  exp_ext::enqueue_signal_event(Q1, Reusable);

  exp_ext::enqueue_wait_event(Q2, Reusable);
  Q2.parallel_for(sycl::range<1>{N}, [=](sycl::id<1> Idx) { Out[Idx] *= 10; });

  Q2.wait();
  Q1.wait();

  std::vector<int> Host(N, -1);
  Q1.memcpy(Host.data(), Out, N * sizeof(int)).wait();
  for (int V : Host)
    assert(V == 10 && "cross-queue event ordering was not respected");

  sycl::free(Out, Q1);
  std::cout << "reusable events: OK" << std::endl;
}

int main() {
  sycl::queue InOrder{sycl::property::queue::in_order{}};
  runOnQueue(InOrder, "in-order");

  sycl::queue OutOfOrder;
  runOnQueue(OutOfOrder, "out-of-order");

  runReusableEventCase();

  return 0;
}
