// Only L0V2 supports urEnqueueHostTaskExp.
// REQUIRES: level_zero_v2_adapter
// REQUIRES: aspect-usm_host_allocations

// UNSUPPORTED: windows && gpu-intel-gen12
// UNSUPPORTED-INTENDED: UR_DEVICE_INFO_ENQUEUE_HOST_TASK_SUPPORT_EXP is not
// supported on win&gen12.

// RUN: %{build} -o %t.out
// RUN: env SYCL_QUEUE_THREAD_POOL_SIZE=8 %{run} %t.out

// tests-10-02 E22: Native host-task ordering remains correct alongside
// scheduler host tasks.
//
// On in-order queue Q1, an ordinary handler::host_task gate holds a kernel
// and a native syclex::host_task, optionally with another ordinary host task
// before or after the native one. Every task records what it saw. Event E is
// signaled after them and captured by a consumer on another queue, then E is
// re-signaled on Q2 and that signal completes. The consumer must keep waiting
// and must finally see the marker of the native task (and of the ordinary
// task). A Q1.wait() on a helper thread must cover both kinds of host work.
// The negative "still pending" checks are best effort.

#include <sycl/detail/core.hpp>
#include <sycl/ext/oneapi/experimental/enqueue_functions.hpp>
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

enum class Ordinary { None, Before, After };

// Markers written by the host work; -1 means not run.
struct Markers {
  int Native = -1;   // the kernel's last element, as seen by the native task
  int Ordinary = -1; // what the ordinary task saw (see runCase)
  int Consumer = -1; // the native marker, as seen by the consumer
};

static void runCase(sycl::queue &Q1, sycl::queue &Q2, sycl::queue &Qc,
                    int *Data, Markers *M, Ordinary Where) {
  const std::string Case = Where == Ordinary::None     ? "native only"
                           : Where == Ordinary::Before ? "ordinary before"
                                                       : "ordinary after";
  Q1.fill(Data, 0, N).wait();
  *M = Markers{};
  sycl::event E = syclex::make_event(Q1.get_context());

  auto G = std::make_shared<Gate>();
  GateOpener Opener{G};
  Q1.submit([&](sycl::handler &CGH) { CGH.host_task([G] { G->wait(); }); });
  Q1.parallel_for(sycl::range<1>(N),
                  [=](sycl::id<1> I) { Data[I] = static_cast<int>(I[0]) + 1; });
  // Before the native task the ordinary task sees the kernel's data; after it,
  // it sees the native task's marker.
  if (Where == Ordinary::Before)
    Q1.submit([&](sycl::handler &CGH) {
      CGH.host_task([=] { M->Ordinary = Data[N - 1]; });
    });
  syclex::host_task(Q1, [=] {
    M->Native = Where == Ordinary::Before ? M->Ordinary : Data[N - 1];
  });
  if (Where == Ordinary::After)
    Q1.submit([&](sycl::handler &CGH) {
      CGH.host_task([=] { M->Ordinary = M->Native; });
    });
  syclex::enqueue_signal_event(Q1, E);

  sycl::event Consumer = Qc.submit([&](sycl::handler &CGH) {
    CGH.depends_on(E);
    CGH.host_task([=] { M->Consumer = M->Native; });
  });

  syclex::enqueue_signal_event(Q2, E);
  E.wait();
  check(isComplete(E), Case, "the new signal is not complete");

  Waiter QueueWait([&] { Q1.wait(); });
  // Best effort: give wrongly released work time to run.
  std::this_thread::sleep_for(100ms);
  check(!isComplete(Consumer), Case, "the consumer did not stay pending");
  check(M->Consumer == -1, Case, "the consumer ran early");
  check(M->Native == -1, Case, "the native task ran before the gate opened");
  check(!QueueWait.done(), Case, "the queue wait did not block");

  G->open();
  QueueWait.finish();
  // The queue wait covers the native and the ordinary host work.
  check(M->Native == static_cast<int>(N), Case,
        "the native task did not see the kernel's data");
  if (Where != Ordinary::None)
    check(M->Ordinary == static_cast<int>(N), Case,
          "the ordinary task did not run in order");
  Consumer.wait();
  check(!G->TimedOut, Case, "the gate timed out");
  check(M->Consumer == static_cast<int>(N), Case,
        "the consumer did not see the native task's marker");
}

int main() {
  sycl::queue Q1{sycl::property::queue::in_order{}};
  sycl::context Ctx = Q1.get_context();
  sycl::device Dev = Q1.get_device();
  sycl::queue Q2{Ctx, Dev, sycl::property::queue::in_order{}};
  sycl::queue Qc{Ctx, Dev, sycl::property::queue::in_order{}};

  int *Data = sycl::malloc_host<int>(N, Q1);
  // Host task markers live in host USM so they outlive every task.
  Markers *M = sycl::malloc_host<Markers>(1, Q1);
  for (Ordinary Where : {Ordinary::None, Ordinary::Before, Ordinary::After})
    runCase(Q1, Q2, Qc, Data, M, Where);

  Q2.wait();
  Qc.wait();
  sycl::free(Data, Q1);
  sycl::free(M, Q1);
  return Failures ? 1 : 0;
}
