// Verifies that SYCL_LAUNCH_BLOCKING makes memory operations synchronous, both
// the ones that return an event and the void-returning free functions, which
// take a different scheduler-bypass path inside the runtime. The destination is
// host-visible, so the check is simply that the data has landed by the time the
// operation returns, without any wait.
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
#include <sycl/ext/oneapi/memcpy2d.hpp>
#include <sycl/properties/queue_properties.hpp>
#include <sycl/usm.hpp>
#include <vector>

namespace exp_ext = sycl::ext::oneapi::experimental;

// Large enough that a copy is still in flight right after an asynchronous
// submission returns.
constexpr size_t Count = 16 * 1024 * 1024;
constexpr size_t Rows = 1024;
constexpr size_t Cols = 1024;

// Set by the RUN lines that enable blocking mode. Without blocking an operation
// may legitimately have completed already, so the checks only apply when this
// is set.
static bool Blocking = false;

static void check(sycl::queue &Q, const char *Dst, char Expected, size_t Len,
                  const char *What) {
  if (Blocking) {
    if (Dst[0] != Expected || Dst[Len - 1] != Expected)
      std::cerr << "result of " << What << " is not visible yet" << std::endl;
    assert(Dst[0] == Expected && Dst[Len - 1] == Expected &&
           "memory operation was not synchronous");
  }
  Q.wait();
}

static void runOnQueue(sycl::queue &Q, const char *Order) {
  // Aligned so that the typed operations below can reuse the same allocations.
  char *Src = sycl::aligned_alloc_shared<char>(sizeof(int), Count, Q);
  char *Dst = sycl::aligned_alloc_shared<char>(sizeof(int), Count, Q);

  // Fills the source with a fresh pattern, so that each check below can only
  // pass if the operation under test has actually run.
  auto prepareSrc = [&](char Value) { Q.memset(Src, Value, Count).wait(); };

  // Event-returning memory operations. These bypass the scheduler in
  // queue_impl::submitMemOpHelper and return a usable event.
  Q.memset(Dst, 1, Count);
  check(Q, Dst, 1, Count, "memset");

  Q.fill(Dst, char{2}, Count);
  check(Q, Dst, 2, Count, "fill");

  prepareSrc(3);
  Q.memcpy(Dst, Src, Count);
  check(Q, Dst, 3, Count, "memcpy");

  prepareSrc(4);
  Q.copy(reinterpret_cast<int *>(Src), reinterpret_cast<int *>(Dst),
         Count / sizeof(int));
  check(Q, Dst, 4, Count, "copy");

  // prefetch and mem_advise take the same two exits of the fast path as the
  // operations above, but have no observable effect on memory, so there is
  // nothing to check for them here.

  // 2D operations are expressed through a command group rather than the
  // memory-operation fast path. memcpy2d stands in for the whole 2D family
  // (fill2d and memset2d take the same route). Its pitches and extents are in
  // bytes, and the pitch matches the width here, so the copied region is
  // contiguous.
  prepareSrc(5);
  Q.ext_oneapi_memcpy2d(Dst, Cols, Src, Cols, Cols, Rows);
  check(Q, Dst, 5, Rows * Cols, "ext_oneapi_memcpy2d");

  // Void-returning free functions. The caller does not need an event, which
  // takes the discard-event branch of the fast path - a different exit from the
  // event-returning calls above.
  exp_ext::memset(Q, Dst, 6, Count);
  check(Q, Dst, 6, Count, "exp::memset");

  exp_ext::fill(Q, Dst, char{7}, Count);
  check(Q, Dst, 7, Count, "exp::fill");

  prepareSrc(8);
  exp_ext::memcpy(Q, Dst, Src, Count);
  check(Q, Dst, 8, Count, "exp::memcpy");

  prepareSrc(9);
  exp_ext::copy(Q, Src, Dst, Count);
  check(Q, Dst, 9, Count, "exp::copy");

  sycl::free(Dst, Q);
  sycl::free(Src, Q);
  std::cout << Order << " queue: OK" << std::endl;
}

// Buffers make the runtime insert its own data-movement commands, which go
// through the scheduler rather than the memory-operation fast path. The kernel
// writes a host-visible flag alongside the buffer to make its completion
// observable without an accessor, which would wait.
static void runBufferCase(sycl::queue &Q) {
  constexpr size_t N = 4096;
  std::vector<int> Data(N, 1);
  int *Done = sycl::malloc_shared<int>(1, Q);
  *Done = 0;
  {
    sycl::buffer<int> Buf{Data.data(), sycl::range<1>{N}};

    Q.submit([&](sycl::handler &CGH) {
      sycl::accessor Acc{Buf, CGH, sycl::read_write};
      CGH.parallel_for(sycl::range<1>{N}, [=](sycl::id<1> Idx) {
        Acc[Idx] = Acc[Idx] * 2;
        if (Idx == N - 1)
          Done[0] = 1;
      });
    });
    if (Blocking && *Done != 1)
      std::cerr << "kernel with buffer accessor has not run yet" << std::endl;
    assert((!Blocking || *Done == 1) && "submission was not synchronous");
    Q.wait();

    sycl::host_accessor HostAcc{Buf, sycl::read_only};
    for (size_t I = 0; I < N; ++I)
      assert(HostAcc[I] == 2 && "buffer kernel produced a wrong result");
  }
  sycl::free(Done, Q);
  std::cout << "buffer case: OK" << std::endl;
}

int main(int argc, char *argv[]) {
  Blocking = argc > 1;

  sycl::queue InOrder{sycl::property::queue::in_order{}};
  runOnQueue(InOrder, "in-order");

  sycl::queue OutOfOrder;
  runOnQueue(OutOfOrder, "out-of-order");

  runBufferCase(InOrder);

  return 0;
}
