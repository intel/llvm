// SYCL_LAUNCH_BLOCKING inserts a queue-wide wait into every submission path.
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
#include <sycl/stream.hpp>
#include <sycl/usm.hpp>
#include <vector>

namespace exp_ext = sycl::ext::oneapi::experimental;

constexpr size_t Rows = 64;
constexpr size_t Cols = 64;
constexpr size_t N = Rows * Cols;

template <typename T> static void check(sycl::queue &Q, const T *Ptr, T Tag) {
  Q.wait();
  assert(Ptr[0] == Tag && Ptr[N - 1] == Tag);
}

static void runKernels(sycl::queue &Q) {
  int *Out = sycl::malloc_shared<int>(N, Q);
  int *Sum = sycl::malloc_shared<int>(1, Q);
  *Sum = 0;
  int Tag = 0;

  // Handler parallel_for: submit_impl.
  ++Tag;
  Q.submit([&](sycl::handler &CGH) {
    CGH.parallel_for(sycl::range<1>{N},
                     [=](sycl::id<1> Idx) { Out[Idx] = Tag; });
  });
  check(Q, Out, Tag);

  // Kernel shortcuts: submit_kernel_direct_impl.
  ++Tag;
  Q.parallel_for(sycl::range<1>{N}, [=](sycl::id<1> Idx) { Out[Idx] = Tag; });
  check(Q, Out, Tag);

  ++Tag;
  Q.single_task([=]() {
    for (size_t I = 0; I < N; ++I)
      Out[I] = Tag;
  });
  check(Q, Out, Tag);

  // Returns no event, taking the discard-event exit of the fast path.
  ++Tag;
  exp_ext::nd_launch(
      Q, sycl::nd_range<1>{sycl::range<1>{N}, sycl::range<1>{64}},
      [=](sycl::nd_item<1> It) { Out[It.get_global_id(0)] = Tag; });
  check(Q, Out, Tag);

  // A reduction submits runtime-internal kernels around the user kernel; the
  // outer submission's wait must cover all of them.
  ++Tag;
  Q.submit([&](sycl::handler &CGH) {
    CGH.parallel_for(sycl::range<1>{N}, sycl::reduction(Sum, sycl::plus<int>()),
                     [=](sycl::id<1> Idx, auto &Reducer) {
                       Out[Idx] = Tag;
                       Reducer += 1;
                     });
  });
  check(Q, Out, Tag);
  assert(*Sum == static_cast<int>(N) && "reduction produced a wrong result");

  // A stream makes submit_impl recurse to submit its flush host task, so the
  // nested submission blocks inside the outer one.
  Q.submit([&](sycl::handler &CGH) {
    sycl::stream OS{1024, 256, CGH};
    CGH.single_task([=]() { OS << 1 << sycl::endl; });
  });
  Q.wait();

  bool HostTaskDone = false;
  Q.submit([&](sycl::handler &CGH) {
    CGH.host_task([&HostTaskDone]() { HostTaskDone = true; });
  });
  Q.wait();
  assert(HostTaskDone && "host task did not run");

  // A kernel depending on a host task cannot bypass the scheduler.
  ++Tag;
  sycl::event HostEvent =
      Q.submit([&](sycl::handler &CGH) { CGH.host_task([]() {}); });
  Q.submit([&](sycl::handler &CGH) {
    CGH.depends_on(HostEvent);
    CGH.parallel_for(sycl::range<1>{N},
                     [=](sycl::id<1> Idx) { Out[Idx] = Tag; });
  });
  check(Q, Out, Tag);

  sycl::free(Sum, Q);
  sycl::free(Out, Q);
}

static void runMemOps(sycl::queue &Q) {
  // Aligned so that the typed operations below can reuse the same allocations.
  char *Src = sycl::aligned_alloc_shared<char>(sizeof(int), N, Q);
  char *Dst = sycl::aligned_alloc_shared<char>(sizeof(int), N, Q);

  // A fresh pattern per operation, so a check can only pass if the operation
  // under test has run.
  auto prepareSrc = [&](char Value) { Q.memset(Src, Value, N).wait(); };

  // Event-returning operations: the scheduler-bypass path of submitMemOpHelper.
  Q.memset(Dst, 1, N);
  check(Q, Dst, char{1});

  Q.fill(Dst, char{2}, N);
  check(Q, Dst, char{2});

  prepareSrc(3);
  Q.memcpy(Dst, Src, N);
  check(Q, Dst, char{3});

  prepareSrc(4);
  Q.copy(reinterpret_cast<int *>(Src), reinterpret_cast<int *>(Dst),
         N / sizeof(int));
  check(Q, Dst, char{4});

  // 2D operations go through a command group instead. memcpy2d stands in for
  // the whole 2D family; its pitches and extents are in bytes and the pitch
  // matches the width, so the region is contiguous.
  prepareSrc(5);
  Q.ext_oneapi_memcpy2d(Dst, Cols, Src, Cols, Cols, Rows);
  check(Q, Dst, char{5});

  // Void-returning free functions: the discard-event exit of the fast path.
  exp_ext::memset(Q, Dst, 6, N);
  check(Q, Dst, char{6});

  exp_ext::fill(Q, Dst, char{7}, N);
  check(Q, Dst, char{7});

  prepareSrc(8);
  exp_ext::memcpy(Q, Dst, Src, N);
  check(Q, Dst, char{8});

  prepareSrc(9);
  exp_ext::copy(Q, Src, Dst, N);
  check(Q, Dst, char{9});

  sycl::free(Dst, Q);
  sycl::free(Src, Q);
}

// Buffers make the runtime insert its own data-movement commands, which go
// through the scheduler rather than the fast path.
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
      assert(HostAcc[I] == 2 && "buffer kernel produced a wrong result");
  }
}

int main() {
  sycl::queue InOrder{sycl::property::queue::in_order{}};
  runKernels(InOrder);
  runMemOps(InOrder);

  sycl::queue OutOfOrder;
  runKernels(OutOfOrder);
  runMemOps(OutOfOrder);

  runBufferCase(InOrder);

  return 0;
}
