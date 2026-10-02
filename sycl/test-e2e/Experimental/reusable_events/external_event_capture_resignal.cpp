// REQUIRES: level_zero_v2_adapter
// REQUIRES: aspect-usm_host_allocations
// RUN: %{build} -o %t.out
// RUN: env SYCL_QUEUE_THREAD_POOL_SIZE=8 %{run} %t.out

// tests-10-02 E16: External-event capture and clearing with reassociation.
//
// Reusable events are used as the external event of an in-order queue
// (ext_oneapi_set_external_event). Each external event is signaled on a
// producer queue behind a host task gate, so that its signal stays pending.
//
// 1. The external event S1 of an idle queue is consumed by the next
//    submission; S1 is then re-signaled elsewhere and that signal completes.
//    The consumer must keep waiting for the original signal of S1.
// 2. The external event is replaced before the next submission: only the most
//    recent one (already complete) is a dependency, so the consumer completes
//    while the replaced one is still pending.
// 3. queue::wait() before any submission waits for the external event and
//    clears it: work submitted afterwards completes while the cleared event
//    has a new pending signal.
//
// The negative "still pending" checks are best effort.

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
#include <thread>

namespace syclex = sycl::ext::oneapi::experimental;
using namespace std::chrono_literals;

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
  void finish() {
    auto Deadline = std::chrono::steady_clock::now() + 60s;
    while (!Done && std::chrono::steady_clock::now() < Deadline)
      std::this_thread::sleep_for(1ms);
    if (!Done) {
      std::cerr << "FAILED: a wait did not return before the deadline\n";
      std::_Exit(1);
    }
    T.join();
  }

private:
  std::atomic<bool> Done{false};
  std::thread T;
};

static int Failures = 0;

static void check(bool Cond, const char *What) {
  if (!Cond) {
    std::cerr << "FAILED: " << What << std::endl;
    ++Failures;
  }
}

static bool isComplete(const sycl::event &E) {
  return E.get_info<sycl::info::event::command_execution_status>() ==
         sycl::info::event_command_status::complete;
}

// Polls E for up to Timeout; true if it completed.
static bool completesWithin(const sycl::event &E,
                            std::chrono::milliseconds Timeout) {
  auto Deadline = std::chrono::steady_clock::now() + Timeout;
  while (!isComplete(E)) {
    if (std::chrono::steady_clock::now() > Deadline)
      return false;
    std::this_thread::sleep_for(1ms);
  }
  return true;
}

// Holds Qp behind a closed gate, writes Value to *Marker and signals S after
// it. Returns the gate.
static std::shared_ptr<Gate> signalBehindGate(sycl::queue &Qp, sycl::event &S,
                                              int *Marker, int Value) {
  auto G = std::make_shared<Gate>();
  Qp.submit([&](sycl::handler &CGH) { CGH.host_task([G] { G->wait(); }); });
  Qp.single_task([=] { *Marker = Value; });
  syclex::enqueue_signal_event(Qp, S);
  return G;
}

int main() {
  sycl::queue Qp{sycl::property::queue::in_order{}};
  sycl::context Ctx = Qp.get_context();
  sycl::device Dev = Qp.get_device();
  sycl::queue Q2{Ctx, Dev, sycl::property::queue::in_order{}};
  sycl::queue Qe{Ctx, Dev, sycl::property::queue::in_order{}};

  int *Marker = sycl::malloc_host<int>(1, Qp);
  int *Out = sycl::malloc_host<int>(1, Qp);

  // 1. The consumer captures S1 when it consumes the external event.
  {
    *Marker = 0;
    *Out = -1;
    sycl::event S1 = syclex::make_event(Ctx);
    GateOpener Opener{signalBehindGate(Qp, S1, Marker, 11)};
    // Qe is idle, so S1 cannot complete before its most recent command.
    Qe.ext_oneapi_set_external_event(S1);
    sycl::event Consumer = Qe.single_task([=] { *Out = *Marker; });

    syclex::enqueue_signal_event(Q2, S1);
    S1.wait();
    check(isComplete(S1), "capture: the new signal is not complete");
    // Best effort: give a wrongly released consumer time to run.
    std::this_thread::sleep_for(100ms);
    check(!isComplete(Consumer), "capture: the consumer did not stay pending");
    check(*Out == -1, "capture: the consumer ran before the producer");

    Opener.G->open();
    Consumer.wait();
    Qp.wait();
    check(!Opener.G->TimedOut, "capture: the gate timed out");
    check(*Out == 11, "capture: the consumer did not see the producer");
  }

  // 2. Only the most recent external event is a dependency.
  {
    *Out = -1;
    sycl::event A = syclex::make_event(Ctx);
    GateOpener Opener{signalBehindGate(Qp, A, Marker, 22)};
    sycl::event B = syclex::make_event(Ctx);
    syclex::enqueue_signal_event(Q2, B);
    B.wait();

    // Qe's most recent command completed in step 1, before B was signaled.
    Qe.ext_oneapi_set_external_event(A);
    Qe.ext_oneapi_set_external_event(B);
    sycl::event Consumer = Qe.single_task([=] { *Out = 2; });
    check(completesWithin(Consumer, 30s),
          "replace: the consumer waited for the replaced external event");
    check(!isComplete(A), "replace: the replaced event completed early");

    Opener.G->open();
    Qp.wait();
    Qe.wait();
    check(!Opener.G->TimedOut, "replace: the gate timed out");
    check(*Out == 2, "replace: the consumer did not run");
    check(*Marker == 22, "replace: the producer did not run");
  }

  // 3. queue::wait() waits for the external event and clears it.
  {
    *Marker = 0;
    *Out = -1;
    sycl::queue Qw{Ctx, Dev, sycl::property::queue::in_order{}};
    sycl::event S = syclex::make_event(Ctx);
    {
      GateOpener Opener{signalBehindGate(Qp, S, Marker, 33)};
      Qw.ext_oneapi_set_external_event(S);
      Waiter QueueWait([&] { Qw.wait(); });
      // Best effort: give a wrongly released wait time to return.
      std::this_thread::sleep_for(100ms);
      check(!QueueWait.done(), "clear: the queue wait did not block");
      Opener.G->open();
      QueueWait.finish();
      check(*Marker == 33, "clear: the queue wait returned before S");
    }

    // Give S a new pending signal: work on Qw must not depend on it any more.
    GateOpener Opener{signalBehindGate(Qp, S, Marker, 44)};
    sycl::event Later = Qw.single_task([=] { *Out = 3; });
    check(completesWithin(Later, 30s),
          "clear: later work inherited the cleared external event");
    check(!isComplete(S), "clear: the new signal of S completed early");

    Opener.G->open();
    Qp.wait();
    Qw.wait();
    check(*Out == 3, "clear: the later work did not run");
    check(*Marker == 44, "clear: the producer did not run");
  }

  Q2.wait();
  Qe.wait();
  sycl::free(Marker, Qp);
  sycl::free(Out, Qp);
  return Failures ? 1 : 0;
}
