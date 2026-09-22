// SYCL_LAUNCH_BLOCKING makes a submission's device work complete before the
// submission returns, so its event is complete and its result readable without
// waiting.
//
// REQUIRES: aspect-usm_shared_allocations
//
// RUN: %{build} -o %t.out
// RUN: env SYCL_LAUNCH_BLOCKING=1 %{run} %t.out

#include <cassert>
#include <sycl/detail/core.hpp>
#include <sycl/ext/oneapi/experimental/enqueue_functions.hpp>
#include <sycl/ext/oneapi/memcpy2d.hpp>
#include <sycl/properties/queue_properties.hpp>
#include <sycl/reduction.hpp>
#include <sycl/usm.hpp>
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

  sycl::free(Sum, Q);
  sycl::free(Out, Q);
}

static void runMemOps(sycl::queue &Q) {
  // Aligned so that the typed operations below can reuse the allocations.
  char *Src = sycl::aligned_alloc_shared<char>(sizeof(int), N, Q);
  char *Dst = sycl::aligned_alloc_shared<char>(sizeof(int), N, Q);
  // A fresh pattern per operation, so a check can only pass if it has run.
  char Tag = 0;
  auto next = [&]() { return ++Tag; };
  auto prepareSrc = [&]() { Q.memset(Src, next(), N); };

  // Event-returning operations: the scheduler-bypass path of submitMemOpHelper.
  checkComplete(Q.memset(Dst, next(), N));
  checkFilled(Dst, Tag);

  checkComplete(Q.fill(Dst, next(), N));
  checkFilled(Dst, Tag);

  prepareSrc();
  checkComplete(Q.memcpy(Dst, Src, N));
  checkFilled(Dst, Tag);

  prepareSrc();
  checkComplete(Q.copy(reinterpret_cast<int *>(Src),
                       reinterpret_cast<int *>(Dst), N / sizeof(int)));
  checkFilled(Dst, Tag);

  // 2D operations go through a command group instead. The pitch matches the
  // width, so the region is contiguous.
  prepareSrc();
  checkComplete(Q.ext_oneapi_memcpy2d(Dst, Cols, Src, Cols, Cols, Rows));
  checkFilled(Dst, Tag);

  // Void-returning free functions: the discard-event exit of the fast path.
  exp_ext::memset(Q, Dst, next(), N);
  checkFilled(Dst, Tag);

  exp_ext::fill(Q, Dst, next(), N);
  checkFilled(Dst, Tag);

  prepareSrc();
  exp_ext::memcpy(Q, Dst, Src, N);
  checkFilled(Dst, Tag);

  prepareSrc();
  exp_ext::copy(Q, Src, Dst, N);
  checkFilled(Dst, Tag);

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
  return 0;
}
