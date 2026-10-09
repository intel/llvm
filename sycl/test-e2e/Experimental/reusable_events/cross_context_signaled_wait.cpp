// REQUIRES: level_zero_v2_adapter
// REQUIRES: aspect-usm_host_allocations

// RUN: %{build} -o %t.out
// RUN: env SYCL_QUEUE_THREAD_POOL_SIZE=8 %{run} %t.out

// tests-10-02 E08 (public waits): A cross-context wait on a deferred signal
// follows the captured signal.
//
// E belongs to context C1 and is signaled on a C1 queue behind a gated
// handler::host_task and a kernel, so the signal is deferred. A queue of a
// second context C2 on the same device waits on E with enqueue_wait_event, or
// with enqueue_wait_events on {E, a C2 reusable event}. A C2 host task and a C2
// kernel then consume the C1 result. E is re-signaled on another C1 queue and
// the new signal completes: the C2 work must still follow the captured signal.
// The handler barrier variant is cross_context_signaled_handler_barrier.cpp.
// Corresponds to unit scenarios U15 and U18.

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

// WithLocalEvent: enqueue_wait_events on {E, a C2 reusable event}.
static void run(bool WithLocalEvent) {
  sycl::device Dev;
  sycl::context C1{Dev};
  sycl::context C2{Dev};
  sycl::queue Q1{C1, Dev, sycl::property::queue::in_order{}};
  sycl::queue Q1b{C1, Dev, sycl::property::queue::in_order{}};
  sycl::queue Q2{C2, Dev, sycl::property::queue::in_order{}};
  sycl::queue Q2b{C2, Dev, sycl::property::queue::in_order{}};

  int *MarkerC1 = sycl::malloc_host<int>(N, C1);
  int *ResC1 = sycl::malloc_host<int>(N, C1);
  int *UnrelatedC1 = sycl::malloc_host<int>(N, C1);
  int *StageC2 = sycl::malloc_host<int>(N, C2);
  int *OutC2 = sycl::malloc_host<int>(N, C2);
  int *LocalC2 = sycl::malloc_host<int>(N, C2);

  auto G = std::make_shared<Gate>();
  OpenOnExit Opener{{G}};

  // A deferred C1 signal.
  Q1.submit([&](sycl::handler &CGH) {
    CGH.host_task([G, MarkerC1] {
      G->pass();
      for (size_t I = 0; I < N; ++I)
        MarkerC1[I] = static_cast<int>(I) + 1;
    });
  });
  G->waitEntered();
  Q1.parallel_for(sycl::range<1>{N},
                  [=](sycl::id<1> I) { ResC1[I] = MarkerC1[I] * 3; });
  sycl::event E = syclex::make_event(C1);
  syclex::enqueue_signal_event(Q1, E);

  // The C2 wait on E.
  if (WithLocalEvent) {
    sycl::event Local = syclex::make_event(C2);
    Q2b.parallel_for(sycl::range<1>{N}, [=](sycl::id<1> I) { LocalC2[I] = 1; });
    syclex::enqueue_signal_event(Q2b, Local);
    syclex::enqueue_wait_events(Q2, {E, Local});
  } else {
    for (size_t I = 0; I < N; ++I)
      LocalC2[I] = 1;
    syclex::enqueue_wait_event(Q2, E);
  }
  Q2.submit([&](sycl::handler &CGH) {
    CGH.host_task([ResC1, StageC2] {
      for (size_t I = 0; I < N; ++I)
        StageC2[I] = ResC1[I];
    });
  });
  sycl::event F = Q2.parallel_for(sycl::range<1>{N}, [=](sycl::id<1> I) {
    OutC2[I] = StageC2[I] + LocalC2[I];
  });

  // Re-signal E in C1 and complete the new signal.
  Q1b.single_task([=] { UnrelatedC1[0] = 9; });
  syclex::enqueue_signal_event(Q1b, E);
  waitBounded(E, "new signal of E");
  std::this_thread::sleep_for(Settle);
  check(!isComplete(F), "C2 work followed the new signal of E (best effort)");

  G->open();
  waitBounded(F, "C2 work after the captured signal completed");
  checkAll(
      OutC2, [](size_t I) { return (static_cast<int>(I) + 1) * 3 + 1; },
      "C2 data");

  drain({Q1, Q1b, Q2, Q2b});
  check(!G->TimedOut, "a host task gate timed out");
  sycl::free(MarkerC1, C1);
  sycl::free(ResC1, C1);
  sycl::free(UnrelatedC1, C1);
  sycl::free(StageC2, C2);
  sycl::free(OutC2, C2);
  sycl::free(LocalC2, C2);
}

int main() {
  try {
    run(/*WithLocalEvent*/ false);
    run(/*WithLocalEvent*/ true);
  } catch (sycl::exception &Ex) {
    std::cerr << "Unexpected exception: " << Ex.what() << std::endl;
    return 1;
  }
  return Failures ? 1 : 0;
}
