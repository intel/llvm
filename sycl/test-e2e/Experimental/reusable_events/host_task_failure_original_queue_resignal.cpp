// REQUIRES: level_zero_v2_adapter
// REQUIRES: aspect-usm_host_allocations
// RUN: %{build} -o %t.out
// RUN: env SYCL_QUEUE_THREAD_POOL_SIZE=8 %{run} %t.out

// tests-10-02 E24: Success and failure of the original work are reported
// against the original queue after a re-signal.
//
// Kernel K on in-order queue Q1 runs behind a host task gate, and its returned
// event E is captured by a consumer on Qc. E is then re-signaled on Q2, and
// that signal completes.
//
// - Success: a Q1.wait() and a Consumer.wait() on helper threads must keep
//   waiting until the gate opens. Then both return and the consumer sees K's
//   data. No async handler is called.
// - Failure: the gate host task throws after the gate opens. The exception
//   must be delivered exactly once, to Q1's async handler, by
//   Q1.wait_and_throw(), which must not return before the gate opens. As for
//   ordinary host task failures, K and the consumer still run. A later signal
//   of E on Q2, used as a dependency on Qc, must not report the failure to
//   Q2's or Qc's async handler.
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
#include <cstring>
#include <iostream>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <string>
#include <thread>

namespace syclex = sycl::ext::oneapi::experimental;
using namespace std::chrono_literals;

constexpr size_t N = 1024;
constexpr const char *FailureMessage = "E24 host task failure";

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

// Exceptions delivered to each queue's async handler.
static std::atomic<int> Q1Exceptions{0};
static std::atomic<int> Q1Expected{0};
static std::atomic<int> OtherExceptions{0};

static void q1Handler(sycl::exception_list List) {
  for (const std::exception_ptr &EP : List) {
    ++Q1Exceptions;
    try {
      std::rethrow_exception(EP);
    } catch (const std::runtime_error &Ex) {
      if (std::strcmp(Ex.what(), FailureMessage) == 0)
        ++Q1Expected;
    } catch (...) {
    }
  }
}

static void otherHandler(sycl::exception_list List) {
  OtherExceptions += static_cast<int>(List.size());
}

static bool kernelDataOk(const int *Data) {
  for (size_t I = 0; I < N; ++I)
    if (Data[I] != static_cast<int>(I) + 1)
      return false;
  return true;
}

static void runCase(sycl::queue &Q1, sycl::queue &Q2, sycl::queue &Qc,
                    int *Data, int *Out, bool Fail) {
  const std::string Case = Fail ? "failure" : "success";
  Q1.fill(Data, 0, N).wait();
  Out[0] = Out[1] = -1;
  const int ExceptionsBefore = Q1Exceptions;

  auto G = std::make_shared<Gate>();
  GateOpener Opener{G};
  Q1.submit([&](sycl::handler &CGH) {
    CGH.host_task([G, Fail] {
      G->wait();
      if (Fail)
        throw std::runtime_error(FailureMessage);
    });
  });
  sycl::event E = Q1.parallel_for(sycl::range<1>(N), [=](sycl::id<1> I) {
    Data[I] = static_cast<int>(I[0]) + 1;
  });
  sycl::event Consumer = Qc.submit([&](sycl::handler &CGH) {
    CGH.depends_on(E);
    CGH.single_task([=] { Out[0] = Data[N - 1]; });
  });

  syclex::enqueue_signal_event(Q2, E);
  E.wait();
  check(isComplete(E), Case, "the new signal is not complete");

  {
    Waiter QueueWait([&] {
      if (Fail)
        Q1.wait_and_throw();
      else
        Q1.wait();
    });
    Waiter ConsumerWait([&] { Consumer.wait(); });
    // Best effort: give wrongly released waits and work time to finish.
    std::this_thread::sleep_for(100ms);
    check(!QueueWait.done(), Case, "the queue wait did not block");
    check(!ConsumerWait.done(), Case, "the consumer wait did not block");
    check(!isComplete(Consumer), Case, "the consumer did not stay pending");
    check(Out[0] == -1, Case, "the consumer ran before the gate opened");
    check(Q1Exceptions == ExceptionsBefore, Case,
          "an exception was reported before the gate opened");

    G->open();
    QueueWait.finish();
    ConsumerWait.finish();
  }
  check(!G->TimedOut, Case, "the gate timed out");
  waitBounded(Consumer);
  check(Out[0] == static_cast<int>(N), Case,
        "the consumer did not see the kernel's data");
  check(kernelDataOk(Data), Case, "wrong kernel data");

  if (Fail) {
    check(Q1Exceptions == ExceptionsBefore + 1, Case,
          "Q1's handler did not get exactly one exception");
    check(Q1Expected == 1, Case, "Q1's handler did not get the failure");
  } else {
    check(Q1Exceptions == ExceptionsBefore, Case,
          "Q1's handler got an exception on success");
  }

  // A later signal of E is an ordinary dependency: the earlier failure is not
  // reported against it.
  syclex::enqueue_signal_event(Q2, E);
  sycl::event Later = Qc.submit([&](sycl::handler &CGH) {
    CGH.depends_on(E);
    CGH.single_task([=] { Out[1] = 1; });
  });
  waitBounded(Later);
  Q2.wait_and_throw();
  Qc.wait_and_throw();
  Q1.wait_and_throw();
  check(Out[1] == 1, Case, "the work after the later signal did not run");
  check(OtherExceptions == 0, Case, "Q2's or Qc's handler got an exception");
  check(Q1Exceptions == ExceptionsBefore + (Fail ? 1 : 0), Case,
        "Q1's handler got an extra exception");
}

int main() {
  sycl::device Dev{sycl::default_selector_v};
  sycl::context Ctx{Dev};
  sycl::queue Q1{Ctx, Dev, q1Handler, sycl::property::queue::in_order{}};
  sycl::queue Q2{Ctx, Dev, otherHandler, sycl::property::queue::in_order{}};
  sycl::queue Qc{Ctx, Dev, otherHandler, sycl::property::queue::in_order{}};

  int *Data = sycl::malloc_host<int>(N, Ctx);
  int *Out = sycl::malloc_host<int>(2, Ctx);
  runCase(Q1, Q2, Qc, Data, Out, /*Fail=*/false);
  runCase(Q1, Q2, Qc, Data, Out, /*Fail=*/true);
  runCase(Q1, Q2, Qc, Data, Out, /*Fail=*/false);

  sycl::free(Data, Ctx);
  sycl::free(Out, Ctx);
  return Failures ? 1 : 0;
}
