// REQUIRES: level_zero_v2_adapter
// REQUIRES: aspect-usm_device_allocations
// RUN: %{build} -o %t.out
// RUN: env SYCL_QUEUE_THREAD_POOL_SIZE=8 %{run} %t.out

// tests-10-02 E12: Memory command chain keeps captured dependencies without
// buffer-induced serialization.
//
// A producer kernel is held behind a host task gate and its returned event E
// is captured by the first command of an explicit chain on an out-of-order
// queue: copy -> transform -> fill -> copy to the host. Every command uses its
// own USM allocation and depends only on the previous one, so no implicit
// buffer dependency can hide a wrong one. E is then re-signaled on another
// queue and that signal completes. The chain must stay pending until the gate
// opens, and must then produce the expected data while the regions it does not
// write stay untouched. A second case does the same with a buffer and a
// sub-buffer, whose accessors add implicit dependencies of their own. The
// negative "still pending" checks are best effort.

#include <sycl/detail/core.hpp>
#include <sycl/ext/oneapi/experimental/reusable_events.hpp>
#include <sycl/properties/all_properties.hpp>
#include <sycl/usm.hpp>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdlib>
#include <iostream>
#include <memory>
#include <mutex>
#include <thread>
#include <vector>

namespace syclex = sycl::ext::oneapi::experimental;
using namespace std::chrono_literals;

constexpr size_t N = 1024;

// A gate an ordinary host task blocks on. The wait is bounded, so a test bug
// fails instead of hanging the suite.
struct Gate {
  std::mutex M;
  std::condition_variable CV;
  bool IsOpen = false;
  std::atomic<bool> TimedOut{false};

  void wait() {
    std::unique_lock<std::mutex> Lock(M);
    if (!CV.wait_for(Lock, 60s, [&] { return IsOpen; }))
      TimedOut = true;
  }
  void open() {
    {
      std::lock_guard<std::mutex> Lock(M);
      IsOpen = true;
    }
    CV.notify_all();
  }
};

// Opens the gate on every exit path.
struct GateOpener {
  std::shared_ptr<Gate> G;
  ~GateOpener() { G->open(); }
};

static int Failures = 0;

static void check(bool Cond, const char *What) {
  if (!Cond) {
    std::cerr << "FAILED: " << What << std::endl;
    ++Failures;
  }
}

static bool isComplete(const sycl::event &E) {
  return E.get_info<sycl::info::event::command_execution_status>() ==
         sycl::info::event_command_status::complete;
}

// Polls E until it completes. A missing completion exits with a failure
// instead of hanging.
static void waitBounded(const sycl::event &E) {
  auto Deadline = std::chrono::steady_clock::now() + 60s;
  while (!isComplete(E)) {
    if (std::chrono::steady_clock::now() > Deadline) {
      std::cerr << "FAILED: no completion before the deadline" << std::endl;
      std::_Exit(1);
    }
    std::this_thread::sleep_for(1ms);
  }
}

// USM chain. Src[i] = i + 1 is written by the gated producer. A has Guard
// ints on both sides of the copied range. B[0, N) is transformed from A,
// B[N, N + Tail) is filled, and B[N + Tail, N + 2 * Tail) is never written.
static void usmChain(sycl::queue &Q1, sycl::queue &Q2, sycl::queue &Qm) {
  constexpr size_t Guard = 64, Tail = 128;
  constexpr int Sentinel = -7, FillValue = 42;
  int *Src = sycl::malloc_device<int>(N, Q1);
  int *A = sycl::malloc_device<int>(N + 2 * Guard, Q1);
  int *B = sycl::malloc_device<int>(N + 2 * Tail, Q1);
  Q1.fill(Src, 0, N);
  Q1.fill(A, Sentinel, N + 2 * Guard);
  Q1.fill(B, Sentinel, N + 2 * Tail);
  Q1.wait();
  std::vector<int> OutA(N + 2 * Guard, 0), OutB(N + 2 * Tail, 0);

  auto G = std::make_shared<Gate>();
  GateOpener Opener{G};
  Q1.submit([&](sycl::handler &CGH) { CGH.host_task([G] { G->wait(); }); });
  sycl::event E = Q1.parallel_for(sycl::range<1>(N), [=](sycl::id<1> I) {
    Src[I] = static_cast<int>(I[0]) + 1;
  });

  // The first command captures the producer's signal.
  sycl::event Copy = Qm.memcpy(A + Guard, Src, N * sizeof(int), E);
  sycl::event Transform = Qm.submit([&](sycl::handler &CGH) {
    CGH.depends_on(Copy);
    CGH.parallel_for(sycl::range<1>(N),
                     [=](sycl::id<1> I) { B[I] = 2 * A[Guard + I[0]]; });
  });
  sycl::event Fill = Qm.fill(B + N, FillValue, Tail, Transform);
  sycl::event Validate = Qm.submit([&](sycl::handler &CGH) {
    CGH.depends_on(Fill);
    CGH.memcpy(OutB.data(), B, OutB.size() * sizeof(int));
  });

  syclex::enqueue_signal_event(Q2, E);
  E.wait();
  check(isComplete(E), "USM chain: the new signal is not complete");

  // Best effort: give a wrongly released chain time to run.
  std::this_thread::sleep_for(100ms);
  check(!isComplete(Copy), "USM chain: the copy did not stay pending");
  check(!isComplete(Validate), "USM chain: the chain did not stay pending");

  G->open();
  waitBounded(Validate);
  Q1.wait();
  Qm.memcpy(OutA.data(), A, OutA.size() * sizeof(int)).wait();
  check(!G->TimedOut, "USM chain: the gate timed out");

  bool OkA = true, OkB = true;
  for (size_t I = 0; I < OutA.size(); ++I) {
    bool Copied = I >= Guard && I < Guard + N;
    OkA &= OutA[I] == (Copied ? static_cast<int>(I - Guard) + 1 : Sentinel);
  }
  for (size_t I = 0; I < OutB.size(); ++I) {
    int Want = I < N ? 2 * (static_cast<int>(I) + 1)
                     : (I < N + Tail ? FillValue : Sentinel);
    OkB &= OutB[I] == Want;
  }
  check(OkA, "USM chain: wrong copied data or guards");
  check(OkB, "USM chain: wrong transformed, filled or untouched data");

  sycl::free(Src, Q1);
  sycl::free(A, Q1);
  sycl::free(B, Q1);
}

// Buffer and sub-buffer chain: the gated producer writes Src, the first
// command copies it into a sub-buffer of Buf, and the second one transforms
// the sub-buffer. The rest of Buf must keep its initial contents.
static void bufferChain(sycl::queue &Q1, sycl::queue &Q2, sycl::queue &Qm) {
  constexpr int Sentinel = -7;
  // A sub-buffer offset has to be aligned to mem_base_addr_align (in bits).
  size_t AlignBits =
      Q1.get_device().get_info<sycl::info::device::mem_base_addr_align>();
  size_t AlignInts = AlignBits / 8 / sizeof(int);
  const size_t Offset = std::max<size_t>(AlignInts, 64);
  std::vector<int> Host(N + 2 * Offset, Sentinel);
  int *Src = sycl::malloc_device<int>(N, Q1);
  Q1.fill(Src, 0, N).wait();

  {
    sycl::buffer<int, 1> Buf{Host.data(), sycl::range<1>(Host.size())};
    sycl::buffer<int, 1> Sub{Buf, sycl::id<1>(Offset), sycl::range<1>(N)};

    auto G = std::make_shared<Gate>();
    GateOpener Opener{G};
    Q1.submit([&](sycl::handler &CGH) { CGH.host_task([G] { G->wait(); }); });
    sycl::event E = Q1.parallel_for(sycl::range<1>(N), [=](sycl::id<1> I) {
      Src[I] = static_cast<int>(I[0]) + 1;
    });

    sycl::event Copy = Qm.submit([&](sycl::handler &CGH) {
      CGH.depends_on(E);
      sycl::accessor Acc{Sub, CGH, sycl::read_write};
      CGH.parallel_for(sycl::range<1>(N),
                       [=](sycl::id<1> I) { Acc[I] = Src[I]; });
    });
    sycl::event Transform = Qm.submit([&](sycl::handler &CGH) {
      sycl::accessor Acc{Sub, CGH, sycl::read_write};
      CGH.parallel_for(sycl::range<1>(N), [=](sycl::id<1> I) { Acc[I] *= 3; });
    });

    syclex::enqueue_signal_event(Q2, E);
    E.wait();
    check(isComplete(E), "buffer chain: the new signal is not complete");

    // Best effort: give a wrongly released chain time to run.
    std::this_thread::sleep_for(100ms);
    check(!isComplete(Copy), "buffer chain: the copy did not stay pending");
    check(!isComplete(Transform),
          "buffer chain: the transform did not stay pending");

    G->open();
    waitBounded(Transform);
    Q1.wait();
    check(!G->TimedOut, "buffer chain: the gate timed out");

    sycl::host_accessor HostAcc{Buf, sycl::read_only};
    bool Ok = true;
    for (size_t I = 0; I < Host.size(); ++I) {
      bool Written = I >= Offset && I < Offset + N;
      Ok &= HostAcc[I] ==
            (Written ? 3 * (static_cast<int>(I - Offset) + 1) : Sentinel);
    }
    check(Ok, "buffer chain: wrong sub-buffer data or untouched regions");
  }

  sycl::free(Src, Q1);
}

int main() {
  sycl::queue Q1{sycl::property::queue::in_order{}};
  sycl::context Ctx = Q1.get_context();
  sycl::device Dev = Q1.get_device();
  sycl::queue Q2{Ctx, Dev, sycl::property::queue::in_order{}};
  // Out-of-order, so the chain is ordered only by its explicit dependencies.
  sycl::queue Qm{Ctx, Dev};

  usmChain(Q1, Q2, Qm);
  bufferChain(Q1, Q2, Qm);

  Q2.wait();
  Qm.wait();
  return Failures ? 1 : 0;
}
