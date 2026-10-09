// REQUIRES: level_zero_v2_adapter
// RUN: %{build} -o %t.out
// RUN: env SYCL_QUEUE_THREAD_POOL_SIZE=8 %{run} %t.out

// tests-10-02 E13: Reduction and buffer destruction retain original resources
// - buffer destruction.
//
// A kernel writing a buffer which copies its data back to the host on
// destruction is held behind a host task gate. Its returned event E is
// re-signaled on another queue and that signal completes. The buffer is then
// destroyed on a helper thread: the destruction must wait for the original
// kernel and copy back its results. review-10-01 #3 notes that the
// scheduler's record cleanup looks at the event's current signal; the wait for
// the release command is expected to hide that here, so this is a guard. The
// "destruction still blocked" check is best effort.

#include <sycl/detail/core.hpp>
#include <sycl/ext/oneapi/experimental/reusable_events.hpp>
#include <sycl/properties/all_properties.hpp>

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
      std::cerr << "FAILED: a call did not return before the deadline\n";
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
  sycl::queue Q2{Q1.get_context(), Q1.get_device(),
                 sycl::property::queue::in_order{}};

  // Repeat, so that buffer allocations and records are reused.
  constexpr int Iterations = 4;
  for (int Iter = 0; Iter < Iterations; ++Iter) {
    std::vector<int> Host(N, -1);
    std::optional<sycl::buffer<int, 1>> Buf{std::in_place, Host.data(),
                                            sycl::range<1>(N)};

    auto G = std::make_shared<Gate>();
    GateOpener Opener{G};
    Q1.submit([&](sycl::handler &CGH) { CGH.host_task([G] { G->wait(); }); });
    std::optional<sycl::event> E = Q1.submit([&](sycl::handler &CGH) {
      sycl::accessor Acc{*Buf, CGH, sycl::read_write};
      CGH.parallel_for(sycl::range<1>(N), [=](sycl::id<1> I) {
        Acc[I] = static_cast<int>(I[0]) * 3 + Iter;
      });
    });

    syclex::enqueue_signal_event(Q2, *E);
    E->wait();
    check(isComplete(*E), Iter, "the new signal is not complete");
    // Drop the public event in every other iteration.
    if (Iter % 2)
      E.reset();

    Waiter Destroy([&] { Buf.reset(); });
    // Best effort: give a wrong destruction time to finish.
    std::this_thread::sleep_for(100ms);
    check(!Destroy.done(), Iter,
          "the buffer destruction did not wait for the original kernel");

    G->open();
    Destroy.finish();
    check(!G->TimedOut, Iter, "the gate timed out");
    bool Ok = true;
    for (size_t I = 0; I < N; ++I)
      Ok &= Host[I] == static_cast<int>(I) * 3 + Iter;
    check(Ok, Iter, "wrong data copied back on buffer destruction");
    Q1.wait();
  }

  Q2.wait();
  return Failures ? 1 : 0;
}
