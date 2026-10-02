// REQUIRES: level_zero_v2_adapter
// REQUIRES: aspect-usm_host_allocations

// RUN: %{build} -o %t.out
// RUN: env SYCL_QUEUE_THREAD_POOL_SIZE=8 %{run} %t.out

// tests-10-02 E03: Independent old and new signals can complete in either
// order.
//
// Every generation of E is signaled on its own in-order queue behind its own
// gated handler::host_task, and captured right away by its own consumer
// (separate queue, separate result storage). The gates are then opened in a
// given order. Each consumer must follow exactly the generation it captured,
// and E and its alias must follow the latest generation. Corresponds to unit
// scenarios U02, U05 and U07.

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

static int markerOf(size_t Gen, size_t I) {
  return static_cast<int>((Gen + 1) * 10000 + I);
}

// ReleaseOrder[k] is the generation whose gate is opened in step k.
static void run(const std::vector<size_t> &ReleaseOrder) {
  const size_t Gens = ReleaseOrder.size();
  sycl::device Dev;
  sycl::context Ctx{Dev};

  std::vector<sycl::queue> Producers, Consumers;
  std::vector<GatePtr> Gates;
  std::vector<int *> Markers, Results;
  std::vector<sycl::event> ConsumerEvents(Gens);
  for (size_t G = 0; G < Gens; ++G) {
    Producers.emplace_back(Ctx, Dev, sycl::property::queue::in_order{});
    Consumers.emplace_back(Ctx, Dev, sycl::property::queue::in_order{});
    Gates.push_back(std::make_shared<Gate>());
    Markers.push_back(sycl::malloc_host<int>(N, Ctx));
    Results.push_back(sycl::malloc_host<int>(N, Ctx));
  }
  OpenOnExit Opener{Gates};

  sycl::event E = syclex::make_event(Ctx);
  sycl::event Alias = E;

  // Hold every producer and confirm all are blocked before signaling.
  for (size_t G = 0; G < Gens; ++G) {
    GatePtr Gt = Gates[G];
    int *M = Markers[G];
    Producers[G].submit([&](sycl::handler &CGH) {
      CGH.host_task([Gt, M, G] {
        Gt->pass();
        for (size_t I = 0; I < N; ++I)
          M[I] = markerOf(G, I);
      });
    });
  }
  for (const GatePtr &Gt : Gates)
    Gt->waitEntered();

  // Generation G: signal E, alternating between E and its alias, and capture
  // the signal in consumer G right away.
  for (size_t G = 0; G < Gens; ++G) {
    syclex::enqueue_signal_event(Producers[G], G % 2 ? Alias : E);
    syclex::enqueue_wait_event(Consumers[G], E);
    const int *M = Markers[G];
    int *R = Results[G];
    ConsumerEvents[G] = Consumers[G].parallel_for(
        sycl::range<1>{N}, [=](sycl::id<1> I) { R[I] = M[I]; });
  }

  std::vector<bool> Released(Gens, false);
  for (size_t Gen : ReleaseOrder) {
    Gates[Gen]->open();
    Released[Gen] = true;

    waitBounded(ConsumerEvents[Gen], "consumer of a released generation");
    checkAll(
        Results[Gen], [Gen](size_t I) { return markerOf(Gen, I); },
        "consumer data of the released generation");

    std::this_thread::sleep_for(Settle);
    for (size_t Other = 0; Other < Gens; ++Other)
      if (!Released[Other])
        check(!isComplete(ConsumerEvents[Other]),
              "consumer of a gated generation completed (best effort)");

    if (Released[Gens - 1]) {
      // No further re-signal follows, so waiting on E is safe here.
      waitBounded(Alias, "E after its latest generation was released");
      check(isComplete(E), "E follows its latest generation");
    } else {
      check(!isComplete(E) && !isComplete(Alias),
            "E completed while its latest generation is gated");
    }
  }

  drain(Producers);
  drain(Consumers);
  for (const GatePtr &Gt : Gates)
    check(!Gt->TimedOut, "a host task gate timed out");
  for (size_t G = 0; G < Gens; ++G) {
    sycl::free(Markers[G], Ctx);
    sycl::free(Results[G], Ctx);
  }
}

int main() {
  try {
    run({1, 0});    // The new signal completes first.
    run({0, 1});    // The old signal completes first.
    run({2, 0, 1}); // Three generations, newest first, then oldest.
  } catch (sycl::exception &Ex) {
    std::cerr << "Unexpected exception: " << Ex.what() << std::endl;
    return 1;
  }
  return Failures ? 1 : 0;
}
