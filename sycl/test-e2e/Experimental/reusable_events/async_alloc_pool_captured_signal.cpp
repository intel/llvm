// REQUIRES: level_zero_v2_adapter
// REQUIRES: aspect-ext_oneapi_async_memory_alloc
// RUN: %{build} -o %t.out
// RUN: env SYCL_QUEUE_THREAD_POOL_SIZE=8 %{run} %t.out

// tests-10-02 E14: Async malloc/free and pools follow captured original work
// - explicit handler dependencies.
//
// An allocation from a memory pool is used by a kernel held behind a host
// task gate; the kernel returns E. A copy of the allocation to the host and
// the async_free of the allocation are submitted through handler APIs with
// explicit dependencies on E (the free also on the copy), then E is
// re-signaled on another queue and that signal completes. The copy and the
// free must keep waiting for the kernel, the copied data must be the
// kernel's, and every queue is drained before the pool is destroyed. The
// pointer is never read after its free. The negative "still pending" checks
// are best effort.
//
// Allocations are made while the queue is idle: an allocation depending on a
// signal still pending in the runtime is the subject of an investigation
// (tests-10-02 U35). Queue helpers inferring the last event are covered by
// async_alloc_helper_last_event_resignal.cpp.

#include <sycl/detail/core.hpp>
#include <sycl/ext/oneapi/experimental/async_alloc/async_alloc.hpp>
#include <sycl/ext/oneapi/experimental/async_alloc/memory_pool.hpp>
#include <sycl/ext/oneapi/experimental/reusable_events.hpp>
#include <sycl/properties/all_properties.hpp>

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdlib>
#include <iostream>
#include <memory>
#include <mutex>
#include <optional>
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

struct DrainOnExit {
  sycl::queue &Q;
  ~DrainOnExit() { Q.wait(); }
};

static int Failures = 0;

static void check(bool Cond, int Gen, const char *What) {
  if (!Cond) {
    std::cerr << "FAILED (generation " << Gen << "): " << What << std::endl;
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

int main() {
  sycl::queue Q1{sycl::property::queue::in_order{}};
  sycl::context Ctx = Q1.get_context();
  sycl::device Dev = Q1.get_device();
  sycl::queue Q2{Ctx, Dev, sycl::property::queue::in_order{}};
  // Out-of-order, so the copy and the free are ordered only by their
  // explicit dependencies.
  sycl::queue Qc{Ctx, Dev};

  {
    syclex::memory_pool Pool(Ctx, Dev, sycl::usm::alloc::device);

    constexpr int Generations = 3;
    for (int Gen = 0; Gen < Generations; ++Gen) {
      int *P = nullptr;
      Q1.submit([&](sycl::handler &CGH) {
        P = static_cast<int *>(
            syclex::async_malloc_from_pool(CGH, N * sizeof(int), Pool));
      });
      Q1.wait();
      std::vector<int> Out(N, -1);
      // On an early exit, the gate opens and the copy into Out finishes
      // before Out is destroyed.
      DrainOnExit DrainCopy{Qc};
      auto G = std::make_shared<Gate>();
      GateOpener Opener{G};
      Q1.submit([&](sycl::handler &CGH) { CGH.host_task([G] { G->wait(); }); });
      std::optional<sycl::event> E =
          Q1.parallel_for(sycl::range<1>(N), [=](sycl::id<1> I) {
            P[I] = static_cast<int>(I[0]) * 7 + Gen;
          });

      sycl::event Copy = Qc.submit([&](sycl::handler &CGH) {
        CGH.depends_on(*E);
        CGH.memcpy(Out.data(), P, N * sizeof(int));
      });
      sycl::event Free = Qc.submit([&](sycl::handler &CGH) {
        CGH.depends_on({*E, Copy});
        syclex::async_free(CGH, P);
      });

      syclex::enqueue_signal_event(Q2, *E);
      E->wait();
      check(isComplete(*E), Gen, "the new signal is not complete");
      // Drop the public event in every other generation.
      if (Gen % 2)
        E.reset();

      // Best effort: give wrongly released operations time to run.
      std::this_thread::sleep_for(100ms);
      check(!isComplete(Copy), Gen, "the copy did not stay pending");
      check(!isComplete(Free), Gen, "the free did not stay pending");

      G->open();
      waitBounded(Copy);
      bool Ok = true;
      for (size_t I = 0; I < N; ++I)
        Ok &= Out[I] == static_cast<int>(I) * 7 + Gen;
      check(Ok, Gen, "the copy did not see the kernel's data");
      waitBounded(Free);
      check(!G->TimedOut, Gen, "the gate timed out");

      Q1.wait();
      Qc.wait();
    }

    // Drain every queue before the pool is destroyed.
    Q1.wait();
    Q2.wait();
    Qc.wait();
  }
  return Failures ? 1 : 0;
}
