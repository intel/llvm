// REQUIRES: level_zero_v2_adapter
// REQUIRES: aspect-usm_host_allocations
// RUN: %{build} -o %t.out
// RUN: env SYCL_QUEUE_THREAD_POOL_SIZE=8 %{run} %t.out

// tests-10-02 E23: Bounded lifetime stress with repeated deferred signals -
// data and progress.
//
// Each batch signals the same event E two or three times on an in-order
// queue. Every signal follows its own host task gate and a kernel writing a
// per-generation marker, so all signals are deferred in the runtime. A
// consumer on an out-of-order queue captures each signal. The gates are then
// opened one at a time: each consumer must complete after its own gate opens,
// see its generation's marker, and the consumers of later generations must
// still be pending. The event is either fresh or kept across batches, and in
// some batches every public alias is dropped before the gates open. Every
// wait is bounded. The negative "still pending" checks are best effort.
//
// This is a guard: review-10-02 #8 (ownership cycles between signals of the
// same event) leaks memory, which these checks cannot observe. Consumers of
// the last signal also break that cycle at the head of the branch. The leak is
// targeted by repeated_deferred_signals_leak.cpp.

#include <sycl/detail/core.hpp>
#include <sycl/ext/oneapi/experimental/reusable_events.hpp>
#include <sycl/properties/all_properties.hpp>
#include <sycl/usm.hpp>

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

constexpr int MaxSignals = 3;

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

// Opens every gate of a batch on every exit path.
struct GatesOpener {
  std::vector<std::shared_ptr<Gate>> &Gates;
  ~GatesOpener() {
    for (auto &G : Gates)
      G->open();
  }
};

static int Failures = 0;

static void check(bool Cond, int Batch, int Signal, const char *What) {
  if (!Cond) {
    std::cerr << "FAILED (batch " << Batch << ", signal " << Signal
              << "): " << What << std::endl;
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

static int markerOf(int Batch, int Signal) { return Batch * 10 + Signal + 1; }

int main() {
  sycl::queue Q1{sycl::property::queue::in_order{}};
  sycl::context Ctx = Q1.get_context();
  sycl::queue Qc{Ctx, Q1.get_device()};

  int *Slot = sycl::malloc_host<int>(MaxSignals, Q1);
  int *Seen = sycl::malloc_host<int>(MaxSignals, Q1);
  // Kept across batches in some of them, so that it is re-signaled after
  // earlier batches completed.
  std::optional<sycl::event> Persistent;

  constexpr int Batches = 12;
  auto Start = std::chrono::steady_clock::now();
  for (int Batch = 0; Batch < Batches; ++Batch) {
    const int Signals = 2 + Batch % 2;
    for (int S = 0; S < MaxSignals; ++S)
      Slot[S] = Seen[S] = -1;

    std::optional<sycl::event> E;
    if (Batch % 4 < 2) {
      if (!Persistent)
        Persistent = syclex::make_event(Ctx);
      E = *Persistent;
    } else {
      E = syclex::make_event(Ctx);
    }

    std::vector<std::shared_ptr<Gate>> Gates;
    GatesOpener Opener{Gates};
    std::vector<sycl::event> Consumers;
    for (int S = 0; S < Signals; ++S) {
      auto G = std::make_shared<Gate>();
      Gates.push_back(G);
      Q1.submit([&](sycl::handler &CGH) { CGH.host_task([G] { G->wait(); }); });
      int Value = markerOf(Batch, S);
      Q1.single_task([=] { Slot[S] = Value; });
      syclex::enqueue_signal_event(Q1, *E);
      Consumers.push_back(Qc.submit([&](sycl::handler &CGH) {
        CGH.depends_on(*E);
        CGH.single_task([=] { Seen[S] = Slot[S]; });
      }));
    }
    // Drop every public alias before the gates open in some batches.
    if (Batch % 3 == 1) {
      E.reset();
      if (Batch % 4 < 2)
        Persistent.reset();
    }

    for (int S = 0; S < Signals; ++S) {
      Gates[S]->open();
      waitBounded(Consumers[S]);
      check(Seen[S] == markerOf(Batch, S), Batch, S,
            "the consumer did not see its generation's marker");
      if (S + 1 < Signals) {
        // Best effort: give wrongly released consumers time to run.
        std::this_thread::sleep_for(20ms);
        for (int Later = S + 1; Later < Signals; ++Later)
          check(!isComplete(Consumers[Later]), Batch, Later,
                "the consumer ran before its gate opened");
      }
    }
    Q1.wait();
    Qc.wait();
    for (int S = 0; S < Signals; ++S)
      check(!Gates[S]->TimedOut, Batch, S, "the gate timed out");
    if (E)
      check(isComplete(*E), Batch, Signals - 1,
            "the last signal is not complete after the drain");
  }

  // Bounded progress for the whole stress run.
  check(std::chrono::steady_clock::now() - Start < 120s, Batches, 0,
        "the batches took too long");

  Persistent.reset();
  sycl::free(Slot, Q1);
  sycl::free(Seen, Q1);
  return Failures ? 1 : 0;
}
