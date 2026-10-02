// REQUIRES: level_zero_v2_adapter
// REQUIRES: aspect-usm_host_allocations

// XFAIL: *
// XFAIL-TRACKER: review-10-02 #5 (enqueue_wait_event and enqueue_wait_events
// still reject every event of a handler::host_task)

// RUN: %{build} -o %t.out
// RUN: env SYCL_QUEUE_THREAD_POOL_SIZE=8 %{run} %t.out

// tests-10-02 E05 (public waits): Ordinary host-task events work with the
// public wait functions.
//
// E is returned by a gated handler::host_task which writes a host marker. Both
// enqueue_wait_event and a one-element enqueue_wait_events must accept E, and
// the commands submitted after them must not run before the host task is
// released. Repeated with E already complete. The handler barrier variants are
// host_task_event_handler_barrier.cpp and deferred_signal_handler_barrier.cpp.
// Corresponds to unit scenario U09.

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

static void run(bool AlreadyComplete) {
  sycl::queue Q1;
  sycl::context Ctx = Q1.get_context();
  sycl::device Dev = Q1.get_device();
  sycl::queue Q2{Ctx, Dev, sycl::property::queue::in_order{}};
  sycl::queue Q3{Ctx, Dev, sycl::property::queue::in_order{}};

  int *Marker = sycl::malloc_host<int>(N, Ctx);
  int *Res1 = sycl::malloc_host<int>(N, Ctx);
  int *Res2 = sycl::malloc_host<int>(N, Ctx);

  auto G = std::make_shared<Gate>();
  OpenOnExit Opener{{G}};

  sycl::event E = Q1.submit([&](sycl::handler &CGH) {
    CGH.host_task([G, Marker] {
      G->pass();
      for (size_t I = 0; I < N; ++I)
        Marker[I] = static_cast<int>(I) + 7;
    });
  });
  if (AlreadyComplete) {
    G->open();
    waitBounded(E, "host task");
  } else {
    G->waitEntered();
  }

  syclex::enqueue_wait_event(Q2, E);
  sycl::event C1 = Q2.parallel_for(
      sycl::range<1>{N}, [=](sycl::id<1> I) { Res1[I] = Marker[I] * 2; });
  syclex::enqueue_wait_events(Q3, {E});
  sycl::event C2 = Q3.parallel_for(
      sycl::range<1>{N}, [=](sycl::id<1> I) { Res2[I] = Marker[I] * 3; });

  if (!AlreadyComplete) {
    std::this_thread::sleep_for(Settle);
    check(!isComplete(C1), "command after enqueue_wait_event ran before the "
                           "host task was released (best effort)");
    check(!isComplete(C2), "command after enqueue_wait_events ran before the "
                           "host task was released (best effort)");
    G->open();
  }

  waitBounded(C1, "command after enqueue_wait_event");
  waitBounded(C2, "command after enqueue_wait_events");
  checkAll(
      Res1, [](size_t I) { return (static_cast<int>(I) + 7) * 2; },
      "data after enqueue_wait_event");
  checkAll(
      Res2, [](size_t I) { return (static_cast<int>(I) + 7) * 3; },
      "data after enqueue_wait_events");

  drain({Q1, Q2, Q3});
  check(!G->TimedOut, "a host task gate timed out");
  for (int *P : {Marker, Res1, Res2})
    sycl::free(P, Ctx);
}

int main() {
  try {
    run(/*AlreadyComplete*/ false);
    run(/*AlreadyComplete*/ true);
  } catch (sycl::exception &Ex) {
    std::cerr << "Unexpected exception: " << Ex.what() << std::endl;
    return 1;
  }
  return Failures ? 1 : 0;
}
