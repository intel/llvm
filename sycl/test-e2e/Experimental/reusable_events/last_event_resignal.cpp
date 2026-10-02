// REQUIRES: level_zero_v2_adapter
// REQUIRES: aspect-usm_host_allocations
// RUN: %{build} -o %t.out
// RUN: env SYCL_QUEUE_THREAD_POOL_SIZE=8 %{run} %t.out

// XFAIL: *
// XFAIL-TRACKER: review-10-02 #7 (ext_oneapi_get_last_event returns the
// wrong signal)

// tests-10-02 E15: Last-event and queue-empty APIs report the original queue's
// work - last event of a re-signaled kernel.
//
// The last kernel K of in-order queue Q1 is held behind a host task gate and
// its returned event E is re-signaled on Q2, where it completes. The event
// returned by Q1.ext_oneapi_get_last_event() must still represent K: it must
// not be complete, and a marker depending on it must not run before K. A
// Q1.wait() on a helper thread must keep waiting too. ext_oneapi_get_last_event
// returns the public event E, now the completed new signal, so the status
// check fails deterministically and the marker runs early. The kernel's
// public event is dropped in the second run. Queue-empty controls and a
// host-only final command are in last_event_queue_empty_resignal.cpp. The
// "still pending" checks are best effort.

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
#include <string>
#include <thread>

namespace syclex = sycl::ext::oneapi::experimental;
using namespace std::chrono_literals;

constexpr size_t N = 1024;

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

static void check(bool Cond, const std::string &Case, const char *What) {
  if (!Cond) {
    std::cerr << "FAILED (" << Case << "): " << What << std::endl;
    ++Failures;
  }
}

static bool isComplete(const sycl::event &E) {
  return E.get_info<sycl::info::event::command_execution_status>() ==
         sycl::info::event_command_status::complete;
}

static void runCase(sycl::queue &Q1, sycl::queue &Q2, sycl::queue &Qm,
                    int *Data, int *Marker, bool DropEvent) {
  const std::string Case =
      DropEvent ? "dropped kernel event" : "kept kernel event";
  Q1.fill(Data, 0, N).wait();
  *Marker = -1;

  auto G = std::make_shared<Gate>();
  GateOpener Opener{G};
  Q1.submit([&](sycl::handler &CGH) { CGH.host_task([G] { G->wait(); }); });
  std::optional<sycl::event> E =
      Q1.parallel_for(sycl::range<1>(N), [=](sycl::id<1> I) {
        Data[I] = static_cast<int>(I[0]) + 1;
      });

  syclex::enqueue_signal_event(Q2, *E);
  E->wait();
  check(isComplete(*E), Case, "the new signal is not complete");
  if (DropEvent)
    E.reset();

  std::optional<sycl::event> Last = Q1.ext_oneapi_get_last_event();
  check(Last.has_value(), Case, "no last event on a nonempty queue");
  if (!Last)
    return;
  check(!isComplete(*Last), Case,
        "the last event is complete while the last kernel is held");

  // The marker copies the kernel's first element.
  sycl::event MarkerEvent = Qm.submit([&](sycl::handler &CGH) {
    CGH.depends_on(*Last);
    CGH.single_task([=] { *Marker = Data[0]; });
  });
  Waiter QueueWait([&] { Q1.wait(); });

  // Best effort: give wrongly released work time to run.
  std::this_thread::sleep_for(100ms);
  check(!isComplete(MarkerEvent), Case, "the marker did not stay pending");
  check(*Marker == -1, Case, "the marker ran before the kernel");
  check(!QueueWait.done(), Case, "the queue wait did not block");

  G->open();
  QueueWait.finish();
  MarkerEvent.wait();
  check(!G->TimedOut, Case, "the gate timed out");
  check(*Marker == 1, Case, "the marker did not see the kernel's data");
  bool Ok = true;
  for (size_t I = 0; I < N; ++I)
    Ok &= Data[I] == static_cast<int>(I) + 1;
  check(Ok, Case, "wrong kernel data");
  check(Q1.ext_oneapi_empty(), Case, "the queue is not empty after a wait");
}

int main() {
  sycl::queue Q1{sycl::property::queue::in_order{}};
  sycl::context Ctx = Q1.get_context();
  sycl::device Dev = Q1.get_device();
  sycl::queue Q2{Ctx, Dev, sycl::property::queue::in_order{}};
  sycl::queue Qm{Ctx, Dev, sycl::property::queue::in_order{}};

  int *Data = sycl::malloc_host<int>(N, Q1);
  int *Marker = sycl::malloc_host<int>(1, Q1);
  runCase(Q1, Q2, Qm, Data, Marker, /*DropEvent=*/false);
  runCase(Q1, Q2, Qm, Data, Marker, /*DropEvent=*/true);

  // A failed case can return with its work still on Q1.
  Q1.wait();
  Q2.wait();
  Qm.wait();
  sycl::free(Data, Q1);
  sycl::free(Marker, Q1);
  return Failures ? 1 : 0;
}
