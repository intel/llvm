// SYCL_LAUNCH_BLOCKING makes a submission's device work complete before the
// submission returns, so its event is complete and its result readable without
// waiting.
//
// REQUIRES: aspect-usm_shared_allocations
//
// RUN: %{build} -o %t.out
// RUN: env SYCL_LAUNCH_BLOCKING=1 %{run} %t.out

#include <cassert>
#include <future>
#include <sycl/detail/core.hpp>
#include <sycl/ext/oneapi/experimental/enqueue_functions.hpp>
#include <sycl/ext/oneapi/memcpy2d.hpp>
#include <sycl/properties/queue_properties.hpp>
#include <sycl/reduction.hpp>
#include <sycl/stream.hpp>
#include <sycl/usm.hpp>
#include <thread>
#include <vector>

namespace exp_ext = sycl::ext::oneapi::experimental;

constexpr size_t Rows = 64;
constexpr size_t Cols = 64;
constexpr size_t N = Rows * Cols;

static bool isComplete(sycl::event E) {
  return E.get_info<sycl::info::event::command_execution_status>() ==
         sycl::info::event_command_status::complete;
}

static void runKernels(sycl::queue &Q) {
  int *Out = sycl::malloc_shared<int>(N, Q);
  int *Sum = sycl::malloc_shared<int>(1, Q);
  *Sum = 0;
  int Tag = 0;

  // Handler submission: submit_impl.
  ++Tag;
  sycl::event E = Q.submit([&](sycl::handler &CGH) {
    CGH.parallel_for(sycl::range<1>{N},
                     [=](sycl::id<1> Idx) { Out[Idx] = Tag; });
  });
  assert(isComplete(E) && Out[0] == Tag && Out[N - 1] == Tag);

  // Kernel shortcut: submit_kernel_direct_impl.
  ++Tag;
  E = Q.parallel_for(sycl::range<1>{N},
                     [=](sycl::id<1> Idx) { Out[Idx] = Tag; });
  assert(isComplete(E) && Out[0] == Tag && Out[N - 1] == Tag);

  ++Tag;
  E = Q.single_task([=]() {
    for (size_t I = 0; I < N; ++I)
      Out[I] = Tag;
  });
  assert(isComplete(E) && Out[0] == Tag && Out[N - 1] == Tag);

  // Returns no event, taking the discard-event exit of the fast path.
  ++Tag;
  exp_ext::nd_launch(
      Q, sycl::nd_range<1>{sycl::range<1>{N}, sycl::range<1>{64}},
      [=](sycl::nd_item<1> It) { Out[It.get_global_id(0)] = Tag; });
  assert(Out[0] == Tag && Out[N - 1] == Tag);

  // A reduction submits runtime-internal kernels around the user kernel.
  ++Tag;
  E = Q.submit([&](sycl::handler &CGH) {
    CGH.parallel_for(sycl::range<1>{N}, sycl::reduction(Sum, sycl::plus<int>()),
                     [=](sycl::id<1> Idx, auto &Reducer) {
                       Out[Idx] = Tag;
                       Reducer += 1;
                     });
  });
  assert(isComplete(E) && Out[0] == Tag && *Sum == static_cast<int>(N));

  // A stream makes submit_impl recurse to submit its flush host task.
  Q.submit([&](sycl::handler &CGH) {
    sycl::stream OS{1024, 256, CGH};
    CGH.single_task([=]() { OS << 1 << sycl::endl; });
  });

  // A kernel depending on a host task cannot bypass the scheduler.
  ++Tag;
  sycl::event HostEvent =
      Q.submit([&](sycl::handler &CGH) { CGH.host_task([]() {}); });
  Q.submit([&](sycl::handler &CGH) {
    CGH.depends_on(HostEvent);
    CGH.parallel_for(sycl::range<1>{N},
                     [=](sycl::id<1> Idx) { Out[Idx] = Tag; });
  });
  Q.wait();
  assert(Out[0] == Tag && Out[N - 1] == Tag);

  sycl::free(Sum, Q);
  sycl::free(Out, Q);
}

static void runMemOps(sycl::queue &Q) {
  // Aligned so that the typed operations below can reuse the allocations.
  char *Src = sycl::aligned_alloc_shared<char>(sizeof(int), N, Q);
  char *Dst = sycl::aligned_alloc_shared<char>(sizeof(int), N, Q);
  // A fresh pattern per operation, so a check can only pass if it has run.
  auto prepareSrc = [&](char Value) { Q.memset(Src, Value, N); };

  // Event-returning operations: the scheduler-bypass path of submitMemOpHelper.
  assert(isComplete(Q.memset(Dst, 1, N)) && Dst[0] == 1 && Dst[N - 1] == 1);
  assert(isComplete(Q.fill(Dst, char{2}, N)) && Dst[0] == 2);

  prepareSrc(3);
  assert(isComplete(Q.memcpy(Dst, Src, N)) && Dst[0] == 3 && Dst[N - 1] == 3);

  prepareSrc(4);
  assert(isComplete(Q.copy(reinterpret_cast<int *>(Src),
                           reinterpret_cast<int *>(Dst), N / sizeof(int))) &&
         Dst[0] == 4);

  // 2D operations go through a command group instead. The pitch matches the
  // width, so the region is contiguous.
  prepareSrc(5);
  assert(isComplete(Q.ext_oneapi_memcpy2d(Dst, Cols, Src, Cols, Cols, Rows)) &&
         Dst[0] == 5 && Dst[N - 1] == 5);

  // Void-returning free functions: the discard-event exit of the fast path.
  exp_ext::memset(Q, Dst, 6, N);
  assert(Dst[0] == 6 && Dst[N - 1] == 6);

  exp_ext::fill(Q, Dst, char{7}, N);
  assert(Dst[0] == 7 && Dst[N - 1] == 7);

  prepareSrc(8);
  exp_ext::memcpy(Q, Dst, Src, N);
  assert(Dst[0] == 8 && Dst[N - 1] == 8);

  prepareSrc(9);
  exp_ext::copy(Q, Src, Dst, N);
  assert(Dst[0] == 9 && Dst[N - 1] == 9);

  sycl::free(Dst, Q);
  sycl::free(Src, Q);
}

// Buffers make the runtime insert its own data-movement commands.
static void runBufferCase(sycl::queue &Q) {
  std::vector<int> Data(N, 1);
  {
    sycl::buffer<int> Buf{Data.data(), sycl::range<1>{N}};
    Q.submit([&](sycl::handler &CGH) {
      sycl::accessor Acc{Buf, CGH, sycl::read_write};
      CGH.parallel_for(sycl::range<1>{N},
                       [=](sycl::id<1> Idx) { Acc[Idx] = Acc[Idx] * 2; });
    });
    sycl::host_accessor HostAcc{Buf, sycl::read_only};
    for (size_t I = 0; I < N; ++I)
      assert(HostAcc[I] == 2);
  }
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

// One thread blocked in a host task must not hold up another thread.
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

  // Submit only once the host task is known to be blocking.
  IsRunning.wait();
  int *Out = sycl::malloc_device<int>(1, Q);
  Q.single_task([=]() { *Out = 1; });

  Gate.set_value();
  Blocked.join();
  Q.wait();
  sycl::free(Out, Q);
}

int main() {
  sycl::queue InOrder{sycl::property::queue::in_order{}};
  runKernels(InOrder);
  runMemOps(InOrder);

  sycl::queue OutOfOrder;
  runKernels(OutOfOrder);
  runMemOps(OutOfOrder);

  runBufferCase(InOrder);
  runGatedHostTask(InOrder);
  runSubmitFromHostTask(InOrder);
  runSharedQueue(OutOfOrder);
  return 0;
}
