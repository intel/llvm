// REQUIRES: level_zero_v2_adapter
// REQUIRES: aspect-usm_device_allocations, aspect-usm_host_allocations

// RUN: %{build} -o %t.out
// RUN: env SYCL_QUEUE_THREAD_POOL_SIZE=8 %{run} %t.out

// tests-10-02 E02: A completed event re-signaled behind a host task stays
// incomplete.
//
// E is re-signaled on an in-order queue behind a gated handler::host_task, so
// the signal is held in the SYCL runtime. Until the gate opens, E must not
// report completion (in particular, the backend event of an earlier, completed
// signal must not stand in for the new one), and neither an E waiter nor a
// consumer waiting on E may finish. Variants: E with a completed earlier
// signal, a fresh E, and E whose completed earlier signal is retained by a
// pending consumer. Real-device counterpart of unit scenario U04.

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

static bool hasReturned(const std::shared_future<void> &Fut) {
  return Fut.wait_for(0s) == std::future_status::ready;
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

enum class Variant { CompletedSignal, Fresh, RetainedCompletedSignal };

static void run(Variant V) {
  sycl::queue Q1{sycl::property::queue::in_order{}};
  sycl::context Ctx = Q1.get_context();
  sycl::device Dev = Q1.get_device();
  sycl::queue Q2{Ctx, Dev, sycl::property::queue::in_order{}};
  sycl::queue Q3{Ctx, Dev, sycl::property::queue::in_order{}};

  int *Marker = sycl::malloc_host<int>(N, Ctx); // Written by the host task.
  int *Mid = sycl::malloc_device<int>(N, Dev, Ctx);
  int *Res = sycl::malloc_host<int>(N, Ctx);
  int *OldRes = sycl::malloc_host<int>(N, Ctx);

  sycl::event E = syclex::make_event(Ctx);
  if (V != Variant::Fresh) {
    Q1.parallel_for(sycl::range<1>{N}, [=](sycl::id<1> I) { Mid[I] = -1; });
    syclex::enqueue_signal_event(Q1, E);
    waitBounded(E, "initial signal");
  }
  check(isComplete(E), "E is complete before it is re-signaled");

  auto GateA = std::make_shared<Gate>();
  auto GateB = std::make_shared<Gate>();
  OpenOnExit Opener{{GateA, GateB}};

  // A pending consumer which retains E's completed signal.
  sycl::event OldConsumer;
  if (V == Variant::RetainedCompletedSignal) {
    Q3.submit(
        [&](sycl::handler &CGH) { CGH.host_task([GateB] { GateB->pass(); }); });
    OldConsumer = Q3.submit([&](sycl::handler &CGH) {
      CGH.depends_on(E);
      CGH.parallel_for(sycl::range<1>{N},
                       [=](sycl::id<1> I) { OldRes[I] = 5; });
    });
    GateB->waitEntered();
  }

  // Re-signal E behind a gated host task and a kernel using its output.
  Q1.submit([&](sycl::handler &CGH) {
    CGH.host_task([GateA, Marker] {
      GateA->pass();
      for (size_t I = 0; I < N; ++I)
        Marker[I] = static_cast<int>(I) + 1;
    });
  });
  GateA->waitEntered();
  Q1.parallel_for(sycl::range<1>{N},
                  [=](sycl::id<1> I) { Mid[I] = Marker[I] * 2; });
  syclex::enqueue_signal_event(Q1, E);

  // Only now, with the reassociation finished, start an E waiter.
  auto WaiterCalling = std::make_shared<std::atomic<bool>>(false);
  sycl::event EWaited = E;
  std::shared_future<void> Waiter =
      startAsync([EWaited, WaiterCalling]() mutable {
        *WaiterCalling = true; // Announced right before the call.
        EWaited.wait();
      });
  // An independent consumer through a wait on E.
  syclex::enqueue_wait_event(Q2, E);
  sycl::event Consumer = Q2.parallel_for(
      sycl::range<1>{N}, [=](sycl::id<1> I) { Res[I] = Mid[I] + 3; });

  while (!*WaiterCalling)
    std::this_thread::sleep_for(1ms);
  std::this_thread::sleep_for(Settle);
  // The new signal cannot have reached the backend yet, so E has to report
  // that it is not complete and the consumer cannot have finished.
  check(!isComplete(E), "re-signaled E reports complete behind a gated host "
                        "task");
  check(!isComplete(Consumer), "consumer of E finished while the host task "
                               "is gated");
  // Best effort: the waiter may not have entered event::wait yet.
  check(!hasReturned(Waiter), "E waiter returned while the host task is gated "
                              "(best effort)");

  if (V == Variant::RetainedCompletedSignal) {
    // The retained signal is complete: its consumer must not follow the new
    // signal, which is still gated.
    GateB->open();
    waitBounded(OldConsumer, "consumer of the retained completed signal");
    checkAll(
        OldRes, [](size_t) { return 5; }, "retained-signal consumer");
    check(!isComplete(E), "E completed with its host task still gated");
  }

  GateA->open();
  finish(Waiter, "E waiter after the gate opened");
  waitBounded(Consumer, "consumer of E after the gate opened");
  check(isComplete(E), "E is complete after its signal finished");
  checkAll(
      Res, [](size_t I) { return (static_cast<int>(I) + 1) * 2 + 3; },
      "consumer data");

  drain({Q1, Q2, Q3});
  check(!GateA->TimedOut && !GateB->TimedOut, "a host task gate timed out");
  sycl::free(Marker, Ctx);
  sycl::free(Mid, Ctx);
  sycl::free(Res, Ctx);
  sycl::free(OldRes, Ctx);
}

int main() {
  try {
    run(Variant::CompletedSignal);
    run(Variant::Fresh);
    run(Variant::RetainedCompletedSignal);
  } catch (sycl::exception &Ex) {
    std::cerr << "Unexpected exception: " << Ex.what() << std::endl;
    return 1;
  }
  return Failures ? 1 : 0;
}
