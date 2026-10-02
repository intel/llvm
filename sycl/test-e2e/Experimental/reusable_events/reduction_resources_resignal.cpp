// REQUIRES: level_zero_v2_adapter
// REQUIRES: aspect-usm_device_allocations
// RUN: %{build} -o %t.out
// RUN: env SYCL_QUEUE_THREAD_POOL_SIZE=8 %{run} %t.out

// XFAIL: *
// XFAIL-TRACKER: review-10-02 #3 (auxiliary resources follow the mutable
// event identity)

// tests-10-02 E13: Reduction and buffer destruction retain original resources
// - reduction temporaries.
//
// A reduction allocates temporary resources (a buffer and its host storage)
// which the scheduler keeps alive, keyed by the event returned by the
// reduction, until that event completes. The reduction is held behind a host
// task gate and its returned event E is captured by a consumer. E is then
// re-signaled on another queue, that signal completes, and an unrelated
// submission lets the scheduler clean up resources.
//
// The cleanup checks E's current signal, so it releases the temporaries of
// the still pending reduction. Destroying the temporary buffer waits for the
// reduction, so the unrelated submission blocks until the gate opens (and may
// deadlock). A premature release cannot be observed directly in an e2e test,
// so this blocking is the deterministic oracle: the submission must return
// while the gate is still closed. The consumer and the queue must still be
// pending at that point, and the sum must be right after the gate opens.

#include <sycl/detail/core.hpp>
#include <sycl/ext/oneapi/experimental/reusable_events.hpp>
#include <sycl/properties/all_properties.hpp>
#include <sycl/reduction.hpp>
#include <sycl/usm.hpp>

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdlib>
#include <functional>
#include <iostream>
#include <memory>
#include <mutex>
#include <thread>

namespace syclex = sycl::ext::oneapi::experimental;
using namespace std::chrono_literals;

constexpr size_t N = 4096;
constexpr size_t WorkGroupSize = 64;

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

// Runs a function on a helper thread. If it does not return before the
// deadline, the process exits with a failure instead of hanging.
class Waiter {
public:
  template <typename F>
  explicit Waiter(F Func)
      : T([this, Func] {
          Func();
          Done = true;
        }) {}
  ~Waiter() {
    if (T.joinable())
      finish();
  }
  bool done() const { return Done; }
  // Waits up to Timeout for the function to return; false if it did not.
  bool doneWithin(std::chrono::milliseconds Timeout) const {
    auto Deadline = std::chrono::steady_clock::now() + Timeout;
    while (!Done && std::chrono::steady_clock::now() < Deadline)
      std::this_thread::sleep_for(1ms);
    return Done;
  }
  void finish() {
    if (!doneWithin(60s)) {
      std::cerr << "FAILED: a call did not return before the deadline (the "
                   "runtime is deadlocked)\n";
      std::_Exit(1);
    }
    T.join();
  }

private:
  std::atomic<bool> Done{false};
  std::thread T;
};

static int Failures = 0;

static void check(bool Cond, int Iter, const char *What) {
  if (!Cond) {
    std::cerr << "FAILED (iteration " << Iter << "): " << What << std::endl;
    ++Failures;
  }
}

static bool isComplete(const sycl::event &E) {
  return E.get_info<sycl::info::event::command_execution_status>() ==
         sycl::info::event_command_status::complete;
}

int main() {
  sycl::queue Q1{sycl::property::queue::in_order{}};
  sycl::context Ctx = Q1.get_context();
  sycl::device Dev = Q1.get_device();
  sycl::queue Q2{Ctx, Dev, sycl::property::queue::in_order{}};
  sycl::queue Qc{Ctx, Dev, sycl::property::queue::in_order{}};
  sycl::queue Qt{Ctx, Dev, sycl::property::queue::in_order{}};

  int *Data = sycl::malloc_device<int>(N, Q1);
  int *Sum = sycl::malloc_device<int>(1, Q1);
  int *Copy = sycl::malloc_device<int>(1, Q1);
  Q1.fill(Data, 1, N).wait();

  // Repeat, so that the resources of earlier iterations are reused.
  constexpr int Iterations = 3;
  for (int Iter = 0; Iter < Iterations; ++Iter) {
    Q1.fill(Sum, 0, 1).wait();
    int Result = -1;

    auto G = std::make_shared<Gate>();
    GateOpener Opener{G};
    Q1.submit([&](sycl::handler &CGH) { CGH.host_task([G] { G->wait(); }); });
    sycl::event E = Q1.parallel_for(
        sycl::nd_range<1>{N, WorkGroupSize},
        sycl::reduction(Sum, std::plus<int>()),
        [=](sycl::nd_item<1> It, auto &S) { S += Data[It.get_global_id(0)]; });
    // The consumer captures the reduction's signal.
    sycl::event Consumer = Qc.submit([&](sycl::handler &CGH) {
      CGH.depends_on(E);
      CGH.single_task([=] { *Copy = *Sum; });
    });

    // Re-signal E, complete the new signal, then make the scheduler clean up
    // resources. This runs on a helper thread because the cleanup can block.
    Waiter Trigger([&] {
      syclex::enqueue_signal_event(Q2, E);
      E.wait();
      Qt.submit([&](sycl::handler &CGH) { CGH.host_task([] {}); }).wait();
    });
    check(Trigger.doneWithin(10s), Iter,
          "an unrelated submission blocked on the pending reduction");
    if (Trigger.done()) {
      check(isComplete(E), Iter, "the new signal is not complete");
      check(!isComplete(Consumer), Iter, "the consumer did not stay pending");
      check(!Q1.ext_oneapi_empty(), Iter, "the original queue became empty");
    }

    G->open();
    Trigger.finish();
    Waiter ConsumerWait([&] {
      Consumer.wait();
      Q1.wait();
      Qc.memcpy(&Result, Copy, sizeof(int)).wait();
    });
    ConsumerWait.finish();
    check(!G->TimedOut, Iter, "the gate timed out");
    check(Result == static_cast<int>(N), Iter, "wrong reduction result");
  }

  Q2.wait();
  Qt.wait();
  sycl::free(Data, Q1);
  sycl::free(Sum, Q1);
  sycl::free(Copy, Q1);
  return Failures ? 1 : 0;
}
