// REQUIRES: level_zero_v2_adapter
// REQUIRES: aspect-usm_host_allocations
// RUN: %{build} -o %t.out
// RUN: env SYCL_QUEUE_THREAD_POOL_SIZE=8 %{run} %t.out %if !gpu || linux %{ | FileCheck %s %}

// tests-10-02 E10: Stream output is complete when original work is waited.
//
// A kernel using sycl::stream is held behind a host task gate. Its returned
// event is captured by a consumer and then re-signaled on an unrelated queue,
// and that new signal completes first. A wait on the original queue must still
// wait for the kernel and for the flush of its stream: the stream line of each
// generation must be printed before the line the host prints when the wait
// returns. The negative "still pending" checks are best effort.

#include <sycl/detail/core.hpp>
#include <sycl/ext/oneapi/experimental/reusable_events.hpp>
#include <sycl/properties/all_properties.hpp>
#include <sycl/stream.hpp>
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

static void check(bool Cond, const char *What, int Gen) {
  if (!Cond) {
    std::cerr << "FAILED (generation " << Gen << "): " << What << std::endl;
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

  int *Marker = sycl::malloc_host<int>(1, Q1);
  constexpr int NumGenerations = 4;

  for (int Gen = 0; Gen < NumGenerations; ++Gen) {
    *Marker = 0;
    auto G = std::make_shared<Gate>();
    GateOpener Opener{G};

    Q1.submit([&](sycl::handler &CGH) { CGH.host_task([G] { G->wait(); }); });
    std::optional<sycl::event> E = Q1.submit([&](sycl::handler &CGH) {
      sycl::stream Out(1024, 80, CGH);
      CGH.single_task([=] { Out << "stream token " << Gen << sycl::endl; });
    });

    // The consumer captures the kernel's signal.
    sycl::event Consumer = Qc.submit([&](sycl::handler &CGH) {
      CGH.depends_on(*E);
      CGH.single_task([=] { *Marker = Gen + 1; });
    });

    // Re-signal the kernel's event on an unrelated queue and complete it.
    syclex::enqueue_signal_event(Q2, *E);
    E->wait();
    check(isComplete(*E), "the new signal is not complete", Gen);
    // Drop the public event in every other generation.
    if (Gen % 2)
      E.reset();

    Waiter OriginalWait([&] { Q1.wait(); });
    // Best effort: give wrongly released work time to run.
    std::this_thread::sleep_for(100ms);
    check(!OriginalWait.done(), "the original queue wait did not block", Gen);
    check(!isComplete(Consumer), "the consumer did not wait for the kernel",
          Gen);
    check(*Marker == 0, "the consumer ran before the kernel", Gen);

    G->open();
    OriginalWait.finish();
    // The stream output must already be flushed: FileCheck verifies that it
    // precedes this line.
    std::cout << "host: generation " << Gen << " waited" << std::endl;

    Consumer.wait();
    check(!G->TimedOut, "the gate timed out", Gen);
    check(*Marker == Gen + 1, "the consumer did not run", Gen);
  }

  // CHECK: stream token 0
  // CHECK-NEXT: host: generation 0 waited
  // CHECK-NEXT: stream token 1
  // CHECK-NEXT: host: generation 1 waited
  // CHECK-NEXT: stream token 2
  // CHECK-NEXT: host: generation 2 waited
  // CHECK-NEXT: stream token 3
  // CHECK-NEXT: host: generation 3 waited

  Q2.wait();
  Qc.wait();
  sycl::free(Marker, Q1);
  return Failures ? 1 : 0;
}
