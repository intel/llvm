// REQUIRES: level_zero_v2_adapter
// REQUIRES: aspect-usm_host_allocations

// RUN: %{build} -o %t.out
// RUN: env SYCL_QUEUE_THREAD_POOL_SIZE=8 %{run} %t.out

// tests-10-02 E04: Signal barrier covers prior work and holds subsequent work.
//
// On an out-of-order queue, a signal of E is a barrier: it covers both earlier
// independent branches (one host-only, one host then device), and the command
// submitted after it must wait for that barrier. Re-signaling E elsewhere, and
// completing the new signal, must not release the follower, nor a consumer
// which captured the original signal. A separate control checks that waiting
// on selected events does not turn into a wait for all previous work.
// Corresponds to unit scenarios U08 and U11.

#include <sycl/detail/core.hpp>
#include <sycl/ext/oneapi/experimental/reusable_events.hpp>
#include <sycl/properties/all_properties.hpp>
#include <sycl/usm.hpp>

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdlib>
#include <exception>
#include <future>
#include <iostream>
#include <memory>
#include <mutex>
#include <thread>
#include <vector>

namespace syclex = sycl::ext::oneapi::experimental;
using namespace std::chrono_literals;

constexpr size_t N = 1024;
// Bound for anything that must finish: a missing wakeup fails, not hangs.
constexpr auto Timeout = 30s;
// Observation window of the best-effort "must still be pending" checks.
constexpr auto Settle = 100ms;

static int Failures = 0;

static void check(bool Cond, const char *What) {
  if (!Cond) {
    std::cerr << "FAILED: " << What << std::endl;
    ++Failures;
  }
}

template <typename F>
static void checkAll(const int *Res, F Expected, const char *What) {
  for (size_t I = 0; I < N; ++I) {
    if (Res[I] != Expected(I)) {
      std::cerr << "FAILED: " << What << ": at " << I << " got " << Res[I]
                << ", expected " << Expected(I) << std::endl;
      ++Failures;
      return;
    }
  }
}

// A thread which may still be blocked cannot be joined: end the process.
[[noreturn]] static void fatal(const char *What) {
  std::cerr << "FATAL: " << What << std::endl;
  std::_Exit(1);
}

// Holds a host task until the test opens it. The host task side is bounded.
struct Gate {
  std::mutex M;
  std::condition_variable CV;
  bool IsOpen = false;
  std::atomic<bool> Entered{false};
  std::atomic<bool> TimedOut{false};

  // Called by the host task.
  void pass() {
    Entered = true;
    std::unique_lock<std::mutex> Lock(M);
    if (!CV.wait_for(Lock, Timeout, [this] { return IsOpen; }))
      TimedOut = true;
  }
  void open() {
    {
      std::lock_guard<std::mutex> Lock(M);
      IsOpen = true;
    }
    CV.notify_all();
  }
  void waitEntered() {
    auto Deadline = std::chrono::steady_clock::now() + Timeout;
    while (!Entered) {
      if (std::chrono::steady_clock::now() > Deadline)
        fatal("a host task did not reach its gate");
      std::this_thread::sleep_for(1ms);
    }
  }
};
using GatePtr = std::shared_ptr<Gate>;

// Opens the gates on every exit path, including exceptions.
struct OpenOnExit {
  std::vector<GatePtr> Gates;
  ~OpenOnExit() {
    for (const GatePtr &G : Gates)
      G->open();
  }
};

// Runs Fn on a detached thread; the future is ready when Fn returns.
template <typename F> static std::shared_future<void> startAsync(F Fn) {
  auto Done = std::make_shared<std::promise<void>>();
  std::shared_future<void> Fut = Done->get_future().share();
  std::thread([Done, Fn]() mutable {
    try {
      Fn();
      Done->set_value();
    } catch (...) {
      Done->set_exception(std::current_exception());
    }
  }).detach();
  return Fut;
}

static void finish(const std::shared_future<void> &Fut, const char *What) {
  if (Fut.wait_for(Timeout) != std::future_status::ready)
    fatal(What);
  Fut.get();
}

static void waitBounded(sycl::event Ev, const char *What) {
  finish(startAsync([Ev]() mutable { Ev.wait(); }), What);
}

static void drain(std::vector<sycl::queue> Queues) {
  finish(startAsync([Queues]() mutable {
           for (sycl::queue &Q : Queues)
             Q.wait();
         }),
         "draining the queues");
}

static bool isComplete(const sycl::event &Ev) {
  return Ev.get_info<sycl::info::event::command_execution_status>() ==
         sycl::info::event_command_status::complete;
}

static void runSignalBarrier() {
  sycl::device Dev;
  sycl::context Ctx{Dev};
  sycl::queue Q{Ctx, Dev}; // Out-of-order.
  sycl::queue QNew{Ctx, Dev, sycl::property::queue::in_order{}};
  sycl::queue QCapture{Ctx, Dev, sycl::property::queue::in_order{}};

  int *MA = sycl::malloc_host<int>(N, Ctx); // Branch A: host task.
  int *MB = sycl::malloc_host<int>(N, Ctx); // Branch B: host task...
  int *DB = sycl::malloc_host<int>(N, Ctx); // ...then a kernel.
  int *Follow = sycl::malloc_host<int>(N, Ctx);
  int *Captured = sycl::malloc_host<int>(N, Ctx);

  auto GateA = std::make_shared<Gate>();
  auto GateB = std::make_shared<Gate>();
  OpenOnExit Opener{{GateA, GateB}};

  sycl::event HA = Q.submit([&](sycl::handler &CGH) {
    CGH.host_task([GateA, MA] {
      GateA->pass();
      for (size_t I = 0; I < N; ++I)
        MA[I] = static_cast<int>(I);
    });
  });
  sycl::event HB = Q.submit([&](sycl::handler &CGH) {
    CGH.host_task([GateB, MB] {
      GateB->pass();
      for (size_t I = 0; I < N; ++I)
        MB[I] = 2;
    });
  });
  Q.submit([&](sycl::handler &CGH) {
    CGH.depends_on(HB);
    CGH.parallel_for(sycl::range<1>{N},
                     [=](sycl::id<1> I) { DB[I] = MB[I] * 10; });
  });
  GateA->waitEntered();
  GateB->waitEntered();

  sycl::event E = syclex::make_event(Ctx);
  syclex::enqueue_signal_event(Q, E);
  // Submitted after the signal barrier: no explicit dependency.
  sycl::event F = Q.parallel_for(
      sycl::range<1>{N}, [=](sycl::id<1> I) { Follow[I] = MA[I] + DB[I]; });
  // Captures the original signal explicitly.
  sycl::event FCaptured = QCapture.submit([&](sycl::handler &CGH) {
    CGH.depends_on(E);
    CGH.parallel_for(sycl::range<1>{N},
                     [=](sycl::id<1> I) { Captured[I] = MA[I] + DB[I]; });
  });

  // Complete branch A only.
  GateA->open();
  waitBounded(HA, "branch A after its gate opened");
  std::this_thread::sleep_for(Settle);
  check(!isComplete(E), "signal completed with branch B still gated");
  check(!isComplete(F), "follower completed with branch B still gated "
                        "(best effort)");
  check(!isComplete(FCaptured), "consumer of the signal completed with branch "
                                "B still gated (best effort)");

  // Re-signal E elsewhere and complete the new signal.
  syclex::enqueue_signal_event(QNew, E);
  waitBounded(E, "new signal of E on an idle queue");
  std::this_thread::sleep_for(Settle);
  check(!isComplete(F), "follower completed after the new signal of E "
                        "(best effort)");
  check(!isComplete(FCaptured), "consumer of the original signal completed "
                                "after the new signal of E (best effort)");

  // Complete the remaining original branch.
  GateB->open();
  waitBounded(F, "follower after both branches completed");
  waitBounded(FCaptured, "consumer of the original signal");
  checkAll(
      Follow, [](size_t I) { return static_cast<int>(I) + 20; },
      "follower sees both branches");
  checkAll(
      Captured, [](size_t I) { return static_cast<int>(I) + 20; },
      "consumer of the original signal sees both branches");

  drain({Q, QNew, QCapture});
  check(!GateA->TimedOut && !GateB->TimedOut, "a host task gate timed out");
  for (int *P : {MA, MB, DB, Follow, Captured})
    sycl::free(P, Ctx);
}

// Waiting on selected events must not become an all-previous-work barrier.
static void runSelectedWaitControl() {
  sycl::device Dev;
  sycl::context Ctx{Dev};
  sycl::queue Q{Ctx, Dev}; // Out-of-order.

  int *Sel = sycl::malloc_host<int>(N, Ctx);
  int *Res = sycl::malloc_host<int>(N, Ctx);
  int *Res2 = sycl::malloc_host<int>(N, Ctx);

  auto Unrelated = std::make_shared<Gate>();
  OpenOnExit Opener{{Unrelated}};

  Q.submit([&](sycl::handler &CGH) {
    CGH.host_task([Unrelated] { Unrelated->pass(); });
  });
  Unrelated->waitEntered();

  sycl::event K = Q.parallel_for(sycl::range<1>{N}, [=](sycl::id<1> I) {
    Sel[I] = static_cast<int>(I) * 7;
  });
  syclex::enqueue_wait_event(Q, K);
  sycl::event F = Q.parallel_for(sycl::range<1>{N},
                                 [=](sycl::id<1> I) { Res[I] = Sel[I] + 1; });
  syclex::enqueue_wait_events(Q, {K, F});
  sycl::event F2 = Q.parallel_for(sycl::range<1>{N},
                                  [=](sycl::id<1> I) { Res2[I] = Res[I] + 1; });

  waitBounded(F2, "work behind waits on selected events, with unrelated work "
                  "gated");
  checkAll(
      Res2, [](size_t I) { return static_cast<int>(I) * 7 + 2; },
      "selected-wait data");

  Unrelated->open();
  drain({Q});
  check(!Unrelated->TimedOut, "a host task gate timed out");
  for (int *P : {Sel, Res, Res2})
    sycl::free(P, Ctx);
}

int main() {
  try {
    runSignalBarrier();
    runSelectedWaitControl();
  } catch (sycl::exception &Ex) {
    std::cerr << "Unexpected exception: " << Ex.what() << std::endl;
    return 1;
  }
  return Failures ? 1 : 0;
}
