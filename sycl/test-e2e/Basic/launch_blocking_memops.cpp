// Verifies that SYCL_LAUNCH_BLOCKING makes memory operations synchronous, both
// the ones that return an event and the void-returning free functions, which
// take a different scheduler-bypass path inside the runtime.
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
#include <sycl/ext/oneapi/memcpy2d.hpp>
#include <sycl/properties/queue_properties.hpp>
#include <sycl/usm.hpp>
#include <vector>

namespace exp_ext = sycl::ext::oneapi::experimental;

// Large enough that a copy is still in flight right after an asynchronous
// submission returns.
constexpr size_t Count = 64 * 1024 * 1024;
constexpr size_t Rows = 1024;
constexpr size_t Cols = 1024;

static const bool Blocking = []() {
  const char *Val = std::getenv("SYCL_LAUNCH_BLOCKING");
  return Val && std::atoi(Val) != 0;
}();

static void checkIdle(sycl::queue &Q, const char *What) {
  const bool Idle = Q.ext_oneapi_empty();
  if (Blocking && !Idle)
    std::cerr << "queue not idle after " << What << std::endl;
  assert((!Blocking || Idle) && "memory operation was not synchronous");
  Q.wait();
}

static void runOnQueue(sycl::queue &Q, const char *Order) {
  char *Src = sycl::malloc_device<char>(Count, Q);
  char *Dst = sycl::malloc_device<char>(Count, Q);
  int *TypedSrc = sycl::malloc_device<int>(Count / sizeof(int), Q);
  int *TypedDst = sycl::malloc_device<int>(Count / sizeof(int), Q);
  int *GridSrc = sycl::malloc_device<int>(Rows * Cols, Q);
  int *GridDst = sycl::malloc_device<int>(Rows * Cols, Q);

  // Event-returning memory operations. These bypass the scheduler in
  // queue_impl::submitMemOpHelper and return a usable event.
  Q.memset(Src, 5, Count);
  checkIdle(Q, "memset");

  Q.fill(Src, char{7}, Count);
  checkIdle(Q, "fill");

  Q.memcpy(Dst, Src, Count);
  checkIdle(Q, "memcpy");

  Q.copy(TypedSrc, TypedDst, Count / sizeof(int));
  checkIdle(Q, "copy");

  // prefetch is deliberately not checked here: it completes too quickly for the
  // check to ever fail, and exp::prefetch below covers the other exit of the
  // fast path.
  Q.mem_advise(Src, Count, 0);
  checkIdle(Q, "mem_advise");

  // 2D operations are expressed through a command group rather than the
  // memory-operation fast path. memcpy2d stands in for the whole 2D family
  // (fill2d and memset2d take the same route).
  Q.ext_oneapi_memcpy2d(GridDst, Cols, GridSrc, Cols, Cols, Rows);
  checkIdle(Q, "ext_oneapi_memcpy2d");

  // Void-returning free functions. The caller does not need an event, which
  // takes the discard-event branch of the fast path - a different exit from the
  // event-returning calls above.
  exp_ext::memset(Q, Src, 1, Count);
  checkIdle(Q, "exp::memset");

  exp_ext::fill(Q, Src, char{2}, Count);
  checkIdle(Q, "exp::fill");

  exp_ext::memcpy(Q, Dst, Src, Count);
  checkIdle(Q, "exp::memcpy");

  exp_ext::copy(Q, Src, Dst, Count);
  checkIdle(Q, "exp::copy");

  exp_ext::prefetch(Q, Src, Count);
  checkIdle(Q, "exp::prefetch");

  // The copy above must have landed; blocking mode does not change results, it
  // only changes when they are visible.
  std::vector<char> Host(1024, 0);
  Q.memcpy(Host.data(), Dst, Host.size()).wait();
  for (char C : Host)
    assert(C == 2 && "copy produced a wrong result");

  sycl::free(GridDst, Q);
  sycl::free(GridSrc, Q);
  sycl::free(TypedDst, Q);
  sycl::free(TypedSrc, Q);
  sycl::free(Dst, Q);
  sycl::free(Src, Q);
  std::cout << Order << " queue: OK" << std::endl;
}

// Buffers make the runtime insert its own data-movement commands, which go
// through the scheduler rather than the memory-operation fast path.
static void runBufferCase(sycl::queue &Q) {
  constexpr size_t N = 4096;
  std::vector<int> Data(N, 1);
  {
    sycl::buffer<int> Buf{Data.data(), sycl::range<1>{N}};

    Q.submit([&](sycl::handler &CGH) {
      sycl::accessor Acc{Buf, CGH, sycl::read_write};
      CGH.parallel_for(sycl::range<1>{N},
                       [=](sycl::id<1> Idx) { Acc[Idx] = Acc[Idx] * 2; });
    });
    checkIdle(Q, "kernel with buffer accessor");

    sycl::host_accessor HostAcc{Buf, sycl::read_only};
    for (size_t I = 0; I < N; ++I)
      assert(HostAcc[I] == 2 && "buffer kernel produced a wrong result");
  }
  std::cout << "buffer case: OK" << std::endl;
}

int main() {
  sycl::queue InOrder{sycl::property::queue::in_order{}};
  runOnQueue(InOrder, "in-order");

  sycl::queue OutOfOrder;
  runOnQueue(OutOfOrder, "out-of-order");

  runBufferCase(InOrder);

  return 0;
}
