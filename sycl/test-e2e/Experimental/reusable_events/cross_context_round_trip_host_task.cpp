// REQUIRES: level_zero_v2_adapter
// REQUIRES: aspect-usm_host_allocations

// RUN: %{build} -o %t.out
// RUN: env SYCL_QUEUE_THREAD_POOL_SIZE=8 %{run} %t.out

// tests-10-02 E09: A cross-context round-trip chain with a host task.
//
// The acyclic chain is
//   C1 gated host task -> C1 kernel -> signal E1 (C1)
//   -> C2 wait on E1 -> C2 host task -> C2 kernel -> signal E2 (C2)
//   -> C1 wait on E2 -> C1 final kernel.
// The whole chain is submitted while the first step is held at a gate; the
// submissions run on a helper thread with a deadline, so a submission blocking
// on the application gate fails instead of hanging. Both intermediate events
// are then re-signaled separately and the new signals complete. Before the gate
// opens, the C2 host task must not have run and the final marker must be unset
// (best effort). After release, every step must see the data of the previous
// one. Device kernels only access storage of their own context; the C2 host
// task copies between contexts through host USM on the host.
// The variant on a backend without native reusable handles needs a different
// adapter and is not covered here. Corresponds to unit scenario U19.

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

static void run() {
  sycl::device Dev;
  sycl::context C1{Dev};
  sycl::context C2{Dev};
  sycl::queue Q1{C1, Dev, sycl::property::queue::in_order{}};
  sycl::queue Q1b{C1, Dev, sycl::property::queue::in_order{}};
  sycl::queue Q1c{C1, Dev, sycl::property::queue::in_order{}};
  sycl::queue Q2{C2, Dev, sycl::property::queue::in_order{}};
  sycl::queue Q2b{C2, Dev, sycl::property::queue::in_order{}};

  int *SrcC1 = sycl::malloc_host<int>(N, C1);
  int *FirstC1 = sycl::malloc_host<int>(N, C1);
  int *MidC1 = sycl::malloc_host<int>(N, C1);
  int *FinalC1 = sycl::malloc_host<int>(N, C1);
  int *UnrelatedC1 = sycl::malloc_host<int>(N, C1);
  int *StageC2 = sycl::malloc_host<int>(N, C2);
  int *OutC2 = sycl::malloc_host<int>(N, C2);
  int *UnrelatedC2 = sycl::malloc_host<int>(N, C2);
  for (size_t I = 0; I < N; ++I) {
    FirstC1[I] = 0;
    MidC1[I] = 0;
    FinalC1[I] = 0;
    StageC2[I] = 0;
    OutC2[I] = 0;
  }
  // Set by the C2 host task: 1 if it saw the complete first step, -1 if not.
  auto HostTaskSaw = std::make_shared<std::atomic<int>>(0);

  auto G = std::make_shared<Gate>();
  OpenOnExit Opener{{G}};

  // Step 1, held at the gate: the source data is written by the gated host
  // task, the C1 kernel transforms it.
  Q1.submit([&](sycl::handler &CGH) {
    CGH.host_task([G, SrcC1] {
      G->pass();
      for (size_t I = 0; I < N; ++I)
        SrcC1[I] = static_cast<int>(I) + 1;
    });
  });
  G->waitEntered();

  sycl::event E1 = syclex::make_event(C1);
  sycl::event E2 = syclex::make_event(C2);
  sycl::event Final;

  // Every downstream submission, on a helper thread with a deadline.
  finish(startAsync([&] {
           Q1.parallel_for(sycl::range<1>{N},
                           [=](sycl::id<1> I) { FirstC1[I] = SrcC1[I] * 2; });
           syclex::enqueue_signal_event(Q1, E1);

           syclex::enqueue_wait_event(Q2, E1);
           Q2.submit([&](sycl::handler &CGH) {
             CGH.host_task([FirstC1, MidC1, StageC2, HostTaskSaw] {
               bool Complete = true;
               for (size_t I = 0; I < N; ++I) {
                 Complete &= FirstC1[I] == (static_cast<int>(I) + 1) * 2;
                 MidC1[I] = FirstC1[I] + 1;
                 StageC2[I] = FirstC1[I];
               }
               HostTaskSaw->store(Complete ? 1 : -1);
             });
           });
           Q2.parallel_for(sycl::range<1>{N},
                           [=](sycl::id<1> I) { OutC2[I] = StageC2[I] * 5; });
           syclex::enqueue_signal_event(Q2, E2);

           syclex::enqueue_wait_event(Q1c, E2);
           Final = Q1c.parallel_for(sycl::range<1>{N}, [=](sycl::id<1> I) {
             FinalC1[I] = MidC1[I] * 3;
           });
         }),
         "submitting the chain blocked on the application gate");

  // Re-signal both captured intermediate events separately and complete the
  // new signals.
  Q1b.single_task([=] { UnrelatedC1[0] = 7; });
  syclex::enqueue_signal_event(Q1b, E1);
  Q2b.single_task([=] { UnrelatedC2[0] = 8; });
  syclex::enqueue_signal_event(Q2b, E2);
  waitBounded(E1, "new signal of E1");
  waitBounded(E2, "new signal of E2");

  std::this_thread::sleep_for(Settle);
  check(HostTaskSaw->load() == 0,
        "the C2 host task ran before the first step was released (best "
        "effort)");
  check(!isComplete(Final),
        "the final step completed before the first step was released (best "
        "effort)");
  check(FinalC1[0] == 0,
        "the final marker was set before the first step was released (best "
        "effort)");

  G->open();
  waitBounded(Final, "final step after the first step was released");
  drain({Q1, Q1b, Q1c, Q2, Q2b});

  check(HostTaskSaw->load() == 1,
        "the C2 host task did not see the complete first step");
  checkAll(
      FirstC1, [](size_t I) { return (static_cast<int>(I) + 1) * 2; },
      "C1 first step");
  checkAll(
      OutC2, [](size_t I) { return (static_cast<int>(I) + 1) * 2 * 5; },
      "C2 kernel");
  checkAll(
      FinalC1, [](size_t I) { return ((static_cast<int>(I) + 1) * 2 + 1) * 3; },
      "C1 final step");
  check(!G->TimedOut, "a host task gate timed out");

  sycl::free(SrcC1, C1);
  sycl::free(FirstC1, C1);
  sycl::free(MidC1, C1);
  sycl::free(FinalC1, C1);
  sycl::free(UnrelatedC1, C1);
  sycl::free(StageC2, C2);
  sycl::free(OutC2, C2);
  sycl::free(UnrelatedC2, C2);
}

int main() {
  try {
    run();
  } catch (sycl::exception &Ex) {
    std::cerr << "Unexpected exception: " << Ex.what() << std::endl;
    return 1;
  }
  return Failures ? 1 : 0;
}
