// REQUIRES: level_zero_v2_adapter
// REQUIRES: aspect-usm_host_allocations
// RUN: %{build} -o %t.out
// RUN: env SYCL_QUEUE_THREAD_POOL_SIZE=8 %{run} %t.out
// RUN: %{l0_leak_check} env SYCL_QUEUE_THREAD_POOL_SIZE=8 %{run} %t.out 2>&1 | FileCheck %s --implicit-check-not=LEAK

// XFAIL: *
// XFAIL-TRACKER: review-10-02 #8 (ownership cycle with repeated deferred
// signals)

// tests-10-02 E23: Bounded lifetime stress with repeated deferred signals -
// release of the runtime objects.
//
// Event E is signaled twice on in-order queue Q1, behind one closed host task
// gate, so both signals are deferred in the scheduler. Nothing depends on the
// last signal. After the gate opens, Q1.wait() drains the queue, and then E,
// the queues and an explicitly created context are dropped. Every runtime
// object must then be released: the Level Zero leak checker must not report a
// leak at exit.
//
// At the head of the branch, the second signal's command keeps E's
// implementation alive and E keeps that command alive. The cycle is broken
// only if a later command depends on the last signal, or if E is signaled
// again. The leaked event holds its context, so the context and its Level Zero
// objects are reported as leaked. This test asserts how the defect should
// show up; the defect has not been observed by running the test. If the
// leak checker does not report it, the test passes unexpectedly
// (XPASS). The first RUN line is the functional control.

#include <sycl/detail/core.hpp>
#include <sycl/ext/oneapi/experimental/reusable_events.hpp>
#include <sycl/properties/all_properties.hpp>
#include <sycl/usm.hpp>

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <iostream>
#include <memory>
#include <mutex>
#include <optional>

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

int main() {
  int Failures = 0;
  {
    sycl::device Dev{sycl::default_selector_v};
    // An explicit context, so that it is released at the end of this scope
    // unless an object leaks.
    sycl::context Ctx{Dev};
    sycl::queue Q1{Ctx, Dev, sycl::property::queue::in_order{}};
    int *Marker = sycl::malloc_host<int>(2, Q1);
    Marker[0] = Marker[1] = -1;

    auto G = std::make_shared<Gate>();
    {
      GateOpener Opener{G};
      std::optional<sycl::event> E = syclex::make_event(Ctx);
      Q1.submit([&](sycl::handler &CGH) { CGH.host_task([G] { G->wait(); }); });
      Q1.single_task([=] { Marker[0] = 1; });
      syclex::enqueue_signal_event(Q1, *E);
      Q1.single_task([=] { Marker[1] = 2; });
      syclex::enqueue_signal_event(Q1, *E);

      G->open();
      // Drain through the queue, not through E.
      Q1.wait();
      E.reset();
    }

    if (G->TimedOut) {
      std::cerr << "FAILED: the gate timed out" << std::endl;
      ++Failures;
    }
    if (Marker[0] != 1 || Marker[1] != 2) {
      std::cerr << "FAILED: the kernels did not run" << std::endl;
      ++Failures;
    }
    sycl::free(Marker, Ctx);
  }
  return Failures ? 1 : 0;
}
