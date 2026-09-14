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

static void checkComplete(sycl::event E) {
  assert(E.get_info<sycl::info::event::command_execution_status>() ==
         sycl::info::event_command_status::complete);
}

template <typename T> static void checkFilled(const T *Ptr, T Value) {
  assert(Ptr[0] == Value && Ptr[N - 1] == Value);
}

// Work long enough that it cannot have finished on its own by the time the
// event is queried, so that this check depends on blocking rather than on
// timing.
static void runLongKernel(sycl::queue &Q) {
  constexpr int Iterations = 5'000'000;
  int *Out = sycl::malloc_shared<int>(1, Q);
  *Out = 0;
  checkComplete(Q.single_task([=]() {
    for (int I = 0; I < Iterations; ++I)
      ++Out[0];
  }));
  assert(*Out == Iterations);
  sycl::free(Out, Q);
}

static void runKernels(sycl::queue &Q) {
  int *Out = sycl::malloc_shared<int>(N, Q);
  int *Sum = sycl::malloc_shared<int>(1, Q);
  *Sum = 0;
  int Tag = 0;

  // Handler submission: submit_impl.
  ++Tag;
  checkComplete(Q.submit([&](sycl::handler &CGH) {
    CGH.parallel_for(sycl::range<1>{N},
                     [=](sycl::id<1> Idx) { Out[Idx] = Tag; });
  }));
  checkFilled(Out, Tag);

  // Kernel shortcut: submit_kernel_direct_impl.
  ++Tag;
  checkComplete(Q.parallel_for(sycl::range<1>{N},
                               [=](sycl::id<1> Idx) { Out[Idx] = Tag; }));
  checkFilled(Out, Tag);

  ++Tag;
  checkComplete(Q.single_task([=]() {
    for (size_t I = 0; I < N; ++I)
      Out[I] = Tag;
  }));
  checkFilled(Out, Tag);

  // Returns no event, taking the discard-event exit of the fast path.
  ++Tag;
  exp_ext::nd_launch(
      Q, sycl::nd_range<1>{sycl::range<1>{N}, sycl::range<1>{64}},
      [=](sycl::nd_item<1> It) { Out[It.get_global_id(0)] = Tag; });
  checkFilled(Out, Tag);

  // A reduction submits runtime-internal kernels around the user kernel.
  ++Tag;
  checkComplete(Q.submit([&](sycl::handler &CGH) {
    CGH.parallel_for(sycl::range<1>{N}, sycl::reduction(Sum, sycl::plus<int>()),
                     [=](sycl::id<1> Idx, auto &Reducer) {
                       Out[Idx] = Tag;
                       Reducer += 1;
                     });
  }));
  checkFilled(Out, Tag);
  assert(*Sum == static_cast<int>(N));

  // A stream makes submit_impl recurse to submit its flush host task.
  Q.submit([&](sycl::handler &CGH) {
    sycl::stream OS{1024, 256, CGH};
    CGH.single_task([=]() { OS << 1 << sycl::endl; });
  });

  // A kernel depending on a host task cannot bypass the scheduler. Host tasks
  // are not made synchronous, hence the wait.
  ++Tag;
  sycl::event HostEvent =
      Q.submit([&](sycl::handler &CGH) { CGH.host_task([]() {}); });
  Q.submit([&](sycl::handler &CGH) {
    CGH.depends_on(HostEvent);
    CGH.parallel_for(sycl::range<1>{N},
                     [=](sycl::id<1> Idx) { Out[Idx] = Tag; });
  });
  Q.wait();
  checkFilled(Out, Tag);

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
  checkComplete(Q.memset(Dst, 1, N));
  checkFilled(Dst, char{1});

  checkComplete(Q.fill(Dst, char{2}, N));
  checkFilled(Dst, char{2});

  prepareSrc(3);
  checkComplete(Q.memcpy(Dst, Src, N));
  checkFilled(Dst, char{3});

  prepareSrc(4);
  checkComplete(Q.copy(reinterpret_cast<int *>(Src),
                       reinterpret_cast<int *>(Dst), N / sizeof(int)));
  checkFilled(Dst, char{4});

  // 2D operations go through a command group instead. The pitch matches the
  // width, so the region is contiguous.
  prepareSrc(5);
  checkComplete(Q.ext_oneapi_memcpy2d(Dst, Cols, Src, Cols, Cols, Rows));
  checkFilled(Dst, char{5});

  // Void-returning free functions: the discard-event exit of the fast path.
  exp_ext::memset(Q, Dst, 6, N);
  checkFilled(Dst, char{6});

  exp_ext::fill(Q, Dst, char{7}, N);
  checkFilled(Dst, char{7});

  prepareSrc(8);
  exp_ext::memcpy(Q, Dst, Src, N);
  checkFilled(Dst, char{8});

  prepareSrc(9);
  exp_ext::copy(Q, Src, Dst, N);
  checkFilled(Dst, char{9});

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

// Blocking never waits for a host task: the three cases below hang if it does.
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
    runLongKernel(*Q);
    runKernels(*Q);
    runMemOps(*Q);
  }

  runBufferCase(InOrder);
  runGatedHostTask(InOrder);
  runSubmitFromHostTask(InOrder);
  runSharedQueue(OutOfOrder);
  return 0;
}
