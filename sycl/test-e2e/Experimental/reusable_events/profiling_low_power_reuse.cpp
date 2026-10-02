// REQUIRES: level_zero_v2_adapter, aspect-usm_device_allocations

// RUN: %{build} -o %t.out
// RUN: env SYCL_QUEUE_THREAD_POOL_SIZE=8 %{run} %t.out

// tests-10-02 E20, completed-signal variants (guard): profiling and low-power
// event properties survive immediate and deferred reuse of an event.
//
// Variants:
//   * per-event profiling: make_event with enable_profiling, on a queue
//     without profiling (only with aspect::ext_oneapi_per_event_profiling);
//   * queue profiling: make_event without properties, signaled on a queue
//     with property::queue::enable_profiling;
//   * low power: make_event with event_mode_enum::low_power, with and without
//     per-event profiling.
// Each variant signals the same event for several generations, immediately
// (the in-order queue is idle) or deferred (behind a gated host task, so the
// signal goes through the scheduler). Profiling is queried only after the
// generation has completed; the pre-completion query is an investigation, in
// profiling_deferred_query.cpp.
//
// Checks: command_start == command_end for the signal tag, command_submit is
// not later than command_start, and the timestamps of a generation are not
// earlier than those of the previous generation on the same device; a deferred
// signal, and a host thread waiting for it, stay pending until the gate opens
// (best effort); the work before the signal is visible after the wait;
// dropping the event while its deferred signal is pending is safe.

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
#include <string>
#include <thread>

namespace syclex = sycl::ext::oneapi::experimental;
using namespace std::chrono_literals;

namespace {

constexpr int Generations = 3;
constexpr auto GateTimeout = 60s;
constexpr auto WaitTimeout = 60s;
// Observation window of the best-effort "still pending" checks.
constexpr auto PendingWindow = 100ms;

int Failures = 0;

#define CHECK(Cond)                                                            \
  do {                                                                         \
    if (!(Cond)) {                                                             \
      std::cerr << __FILE__ << ":" << __LINE__ << ": check failed: " #Cond     \
                << std::endl;                                                  \
      ++Failures;                                                              \
    }                                                                          \
  } while (0)

struct Gate {
  std::mutex M;
  std::condition_variable CV;
  bool IsOpen = true;
  std::atomic<int> Entered{0};
  std::atomic<bool> TimedOut{false};

  void open() {
    {
      std::lock_guard<std::mutex> L(M);
      IsOpen = true;
    }
    CV.notify_all();
  }
  // Only called while no host task is inside pass().
  void close() {
    std::lock_guard<std::mutex> L(M);
    IsOpen = false;
  }
  void pass() {
    ++Entered;
    std::unique_lock<std::mutex> L(M);
    if (!CV.wait_for(L, GateTimeout, [this] { return IsOpen; }))
      TimedOut = true;
  }
};

Gate QueueGate;

struct GateOpener {
  ~GateOpener() { QueueGate.open(); }
};

[[noreturn]] void fatal(const std::string &Msg) {
  std::cerr << "FATAL: " << Msg << std::endl;
  QueueGate.open();
  std::_Exit(1);
}

template <typename Pred>
bool waitUntil(Pred P, std::chrono::milliseconds Timeout) {
  auto Deadline = std::chrono::steady_clock::now() + Timeout;
  while (!P()) {
    if (std::chrono::steady_clock::now() > Deadline)
      return P();
    std::this_thread::sleep_for(1ms);
  }
  return true;
}

bool isComplete(const sycl::event &E) {
  return E.get_info<sycl::info::event::command_execution_status>() ==
         sycl::info::event_command_status::complete;
}

// A host thread blocked in event::wait, joined with a deadline.
struct Waiter {
  std::shared_ptr<std::atomic<bool>> Done =
      std::make_shared<std::atomic<bool>>(false);
  std::shared_ptr<std::string> Error = std::make_shared<std::string>();
  std::thread T;

  explicit Waiter(sycl::event E) {
    T = std::thread([E, Done = Done, Error = Error]() mutable {
      try {
        E.wait();
      } catch (const sycl::exception &Ex) {
        *Error = Ex.what();
      }
      Done->store(true);
    });
  }
  void join(const std::string &What) {
    if (!waitUntil([&] { return Done->load(); }, WaitTimeout)) {
      T.detach();
      fatal("wait timed out: " + What);
    }
    T.join();
    if (!Error->empty()) {
      std::cerr << "wait failed: " << What << ": " << *Error << std::endl;
      ++Failures;
    }
  }
};

void boundedWait(sycl::event E, const std::string &What) {
  Waiter W(E);
  W.join(What);
}

enum class Kind {
  PerEventProfiling,
  QueueProfiling,
  LowPower,
  LowPowerProfiling
};

const char *name(Kind K) {
  switch (K) {
  case Kind::PerEventProfiling:
    return "per-event profiling";
  case Kind::QueueProfiling:
    return "queue profiling";
  case Kind::LowPower:
    return "low power";
  case Kind::LowPowerProfiling:
    return "low power with per-event profiling";
  }
  return "";
}

sycl::event makeEvent(const sycl::context &Ctx, Kind K) {
  switch (K) {
  case Kind::PerEventProfiling:
    return syclex::make_event(
        Ctx, syclex::properties{syclex::enable_profiling{true}});
  case Kind::QueueProfiling:
    return syclex::make_event(Ctx);
  case Kind::LowPower:
    return syclex::make_event(Ctx, syclex::properties{syclex::event_mode{
                                       syclex::event_mode_enum::low_power}});
  case Kind::LowPowerProfiling:
    return syclex::make_event(
        Ctx, syclex::properties{
                 syclex::event_mode{syclex::event_mode_enum::low_power},
                 syclex::enable_profiling{true}});
  }
  return syclex::make_event(Ctx);
}

bool profiled(Kind K) { return K != Kind::LowPower; }

// Enqueues a host task passing QueueGate on Q, and waits until it runs.
void blockQueue(sycl::queue &Q) {
  QueueGate.close();
  const int Base = QueueGate.Entered.load();
  Q.submit(
      [&](sycl::handler &CGH) { CGH.host_task([]() { QueueGate.pass(); }); });
  if (!waitUntil([&] { return QueueGate.Entered.load() == Base + 1; },
                 WaitTimeout))
    fatal("host task did not start");
}

void runVariant(sycl::context &Ctx, sycl::device &Dev, Kind K, bool Deferred) {
  std::cout << "Variant: " << name(K) << ", "
            << (Deferred ? "deferred" : "immediate") << " reuse" << std::endl;
  GateOpener Opener;

  sycl::property_list Props{sycl::property::queue::in_order{}};
  if (K == Kind::QueueProfiling)
    Props = sycl::property_list{sycl::property::queue::in_order{},
                                sycl::property::queue::enable_profiling{}};
  sycl::queue Q{Ctx, Dev, Props};

  int *Data = sycl::malloc_device<int>(1, Q);
  Q.memset(Data, 0, sizeof(int)).wait();

  sycl::event E = makeEvent(Ctx, K);
  uint64_t PrevEnd = 0;
  for (int Gen = 1; Gen <= Generations; ++Gen) {
    // The first generation is always immediate, so that the deferred ones
    // reuse an event which already has a backend event.
    const bool Defer = Deferred && Gen > 1;
    if (Defer)
      blockQueue(Q);
    Q.single_task([=]() { *Data = Gen; });
    syclex::enqueue_signal_event(Q, E);

    if (Defer) {
      // Best effort: the signal and a waiting thread cannot complete while
      // the host task is gated.
      Waiter W(E);
      std::this_thread::sleep_for(PendingWindow);
      CHECK(!isComplete(E));
      CHECK(!W.Done->load());
      QueueGate.open();
      W.join("deferred signal");
    } else {
      boundedWait(E, "immediate signal");
    }
    CHECK(isComplete(E));

    int Host = 0;
    Q.memcpy(&Host, Data, sizeof(int)).wait();
    CHECK(Host == Gen);

    if (profiled(K)) {
      using namespace sycl::info;
      const uint64_t Submit =
          E.get_profiling_info<event_profiling::command_submit>();
      const uint64_t Start =
          E.get_profiling_info<event_profiling::command_start>();
      const uint64_t End = E.get_profiling_info<event_profiling::command_end>();
      CHECK(Start == End);
      CHECK(Submit <= Start);
      CHECK(End != 0);
      CHECK(End >= PrevEnd);
      PrevEnd = End;
    }
  }

  // Drop an event while its deferred signal is pending.
  {
    sycl::event Dropped = makeEvent(Ctx, K);
    syclex::enqueue_signal_event(Q, Dropped);
    boundedWait(Dropped, "first signal of the dropped event");
    blockQueue(Q);
    syclex::enqueue_signal_event(Q, Dropped);
  }
  QueueGate.open();
  Q.wait_and_throw();

  sycl::free(Data, Q);
}

} // namespace

int main() {
  GateOpener Opener;
  try {
    sycl::device Dev;
    sycl::context Ctx{Dev};
    const bool PerEvent = Dev.has(sycl::aspect::ext_oneapi_per_event_profiling);
    if (!PerEvent)
      std::cout << "No aspect::ext_oneapi_per_event_profiling, per-event "
                   "profiling variants skipped"
                << std::endl;
    for (Kind K : {Kind::PerEventProfiling, Kind::QueueProfiling,
                   Kind::LowPower, Kind::LowPowerProfiling}) {
      if (!PerEvent &&
          (K == Kind::PerEventProfiling || K == Kind::LowPowerProfiling))
        continue;
      for (bool Deferred : {false, true})
        runVariant(Ctx, Dev, K, Deferred);
    }
  } catch (const sycl::exception &Ex) {
    std::cerr << "Unexpected exception: " << Ex.what() << std::endl;
    return 1;
  }
  CHECK(!QueueGate.TimedOut.load());
  if (Failures) {
    std::cerr << Failures << " check(s) failed" << std::endl;
    return 1;
  }
  return 0;
}
