// REQUIRES: level_zero_v2_adapter
// REQUIRES: aspect-usm_device_allocations, aspect-usm_host_allocations

// XFAIL: *
// XFAIL-TRACKER: review-10-02 #6 (handler barrier wait lists do not schedule
// a pending reusable signal: a deferred signal is passed on as a null or stale
// backend handle)

// RUN: %{build} -o %t.out
// RUN: env SYCL_QUEUE_THREAD_POOL_SIZE=8 %{run} %t.out

// tests-10-02 E05 (handler barrier, deferred reusable signal): A handler
// barrier on a reusable event whose signal is still held in the SYCL runtime.
//
// E is signaled on an in-order queue behind a gated handler::host_task and a
// kernel, so the signal is deferred. handler::ext_oneapi_barrier({E}) on an
// in-order queue, and a vector barrier mixing E with a kernel event on an
// out-of-order queue, must hold the commands submitted after them until the
// captured signal completes. E is then re-signaled elsewhere and the new signal
// completes: the followers must still wait for the captured signal. The
// barrier submissions are bounded, so a blocking submission fails instead of
// hanging. Corresponds to unit scenario U15.

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

// WithOtherEvent: an out-of-order queue and a barrier on {E, K}.
static void run(bool WithOtherEvent) {
  sycl::queue Q1{sycl::property::queue::in_order{}};
  sycl::context Ctx = Q1.get_context();
  sycl::device Dev = Q1.get_device();
  sycl::queue Q2 =
      WithOtherEvent ? sycl::queue{Ctx, Dev}
                     : sycl::queue{Ctx, Dev, sycl::property::queue::in_order{}};
  sycl::queue Q3{Ctx, Dev, sycl::property::queue::in_order{}};

  int *Marker = sycl::malloc_host<int>(N, Ctx);
  int *Mid = sycl::malloc_device<int>(N, Dev, Ctx);
  int *Aux = sycl::malloc_host<int>(N, Ctx);
  int *Unrelated = sycl::malloc_device<int>(N, Dev, Ctx);
  int *Res = sycl::malloc_host<int>(N, Ctx);

  auto G = std::make_shared<Gate>();
  OpenOnExit Opener{{G}};

  // A deferred signal of E: held behind a gated host task and a kernel.
  Q1.submit([&](sycl::handler &CGH) {
    CGH.host_task([G, Marker] {
      G->pass();
      for (size_t I = 0; I < N; ++I)
        Marker[I] = static_cast<int>(I) + 1;
    });
  });
  G->waitEntered();
  Q1.parallel_for(sycl::range<1>{N},
                  [=](sycl::id<1> I) { Mid[I] = Marker[I] * 2; });
  sycl::event E = syclex::make_event(Ctx);
  syclex::enqueue_signal_event(Q1, E);

  std::vector<sycl::event> WaitList{E};
  if (WithOtherEvent) {
    sycl::event K =
        Q2.parallel_for(sycl::range<1>{N}, [=](sycl::id<1> I) { Aux[I] = 1; });
    WaitList.push_back(K);
  } else {
    for (size_t I = 0; I < N; ++I)
      Aux[I] = 1;
  }

  // The barrier submission must not wait for the gated signal.
  finish(startAsync([Q2, WaitList]() mutable {
           Q2.submit(
               [&](sycl::handler &CGH) { CGH.ext_oneapi_barrier(WaitList); });
         }),
         "submission of a handler barrier on a deferred signal");
  sycl::event Follower = Q2.parallel_for(
      sycl::range<1>{N}, [=](sycl::id<1> I) { Res[I] = Mid[I] + Aux[I]; });

  std::this_thread::sleep_for(Settle);
  check(!isComplete(Follower), "follower of a barrier on a deferred signal ran "
                               "before the signal (best effort)");

  // Re-signal E after the barrier captured it, and complete the new signal.
  Q3.parallel_for(sycl::range<1>{N}, [=](sycl::id<1> I) { Unrelated[I] = 9; });
  syclex::enqueue_signal_event(Q3, E);
  waitBounded(E, "new signal of E");
  std::this_thread::sleep_for(Settle);
  check(!isComplete(Follower), "follower of a barrier on a deferred signal "
                               "followed the new signal of E (best effort)");

  G->open();
  waitBounded(Follower, "follower after the captured signal completed");
  checkAll(
      Res, [](size_t I) { return (static_cast<int>(I) + 1) * 2 + 1; },
      "follower data");

  drain({Q1, Q2, Q3});
  check(!G->TimedOut, "a host task gate timed out");
  for (int *P : {Marker, Mid, Aux, Unrelated, Res})
    sycl::free(P, Ctx);
}

int main() {
  try {
    run(/*WithOtherEvent*/ false);
    run(/*WithOtherEvent*/ true);
  } catch (sycl::exception &Ex) {
    std::cerr << "Unexpected exception: " << Ex.what() << std::endl;
    return 1;
  }
  return Failures ? 1 : 0;
}
