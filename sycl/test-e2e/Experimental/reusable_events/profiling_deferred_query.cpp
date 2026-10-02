// REQUIRES: level_zero_v2_adapter

// RUN: %{build} -o %t.out
// RUN: env SYCL_QUEUE_THREAD_POOL_SIZE=8 %{run} %t.out

// Investigation (tests-10-02 E20): pins behaviour at 8337da70; oracle not
// agreed.
//
// tests-10-02 E20, pre-completion variant: profiling queries on a reused
// event while its new signal is deferred (behind a gated host task) and has
// no backend event yet.
//
// Pinned at 8337da70: every signal of a reusable event is a profiling tag
// event, and event::get_profiling_info (event.cpp) calls impl->wait() before
// command_start and command_end, and before command_submit of a tag event.
// The three queries therefore block until the deferred signal completes, and
// then return the new generation's tag timestamp: submit == start == end,
// non-zero, and not earlier than the previous generation's timestamp. They do
// not return the previous generation's timestamp, nor the binding's
// MSubmitTime (0), which event_impl::get_profiling_info returns without a
// backend event.
//
// The queries run on controlled threads that announce the call before they
// make it. "Still blocked while the gate is closed" is best effort (bounded
// observation window); the values returned after the gate opens are
// deterministic. The event is never re-signaled while a query thread runs.
//
// Variants: per-event profiling (make_event with enable_profiling, only with
// aspect::ext_oneapi_per_event_profiling) and queue profiling (a plain event
// signaled on a queue with enable_profiling), each on an in-order and an
// out-of-order queue.

#include <sycl/detail/core.hpp>
#include <sycl/ext/oneapi/experimental/reusable_events.hpp>
#include <sycl/properties/all_properties.hpp>

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdint>
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
// Observation window of the best-effort "query is still blocked" check.
constexpr auto ObservationWindow = 200ms;

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

void boundedWait(sycl::event E, const std::string &What) {
  auto Done = std::make_shared<std::atomic<bool>>(false);
  std::thread T([E, Done]() mutable {
    try {
      E.wait();
    } catch (const sycl::exception &Ex) {
      std::cerr << "wait threw: " << Ex.what() << std::endl;
    }
    Done->store(true);
  });
  if (!waitUntil([&] { return Done->load(); }, WaitTimeout)) {
    T.detach();
    fatal("wait timed out: " + What);
  }
  T.join();
}

enum class Query { Submit, Start, End };

const char *name(Query Q) {
  switch (Q) {
  case Query::Submit:
    return "command_submit";
  case Query::Start:
    return "command_start";
  case Query::End:
    return "command_end";
  }
  return "";
}

uint64_t query(const sycl::event &E, Query Q) {
  using namespace sycl::info;
  switch (Q) {
  case Query::Submit:
    return E.get_profiling_info<event_profiling::command_submit>();
  case Query::Start:
    return E.get_profiling_info<event_profiling::command_start>();
  case Query::End:
    return E.get_profiling_info<event_profiling::command_end>();
  }
  return 0;
}

// A profiling query on a controlled thread.
struct QueryThread {
  std::shared_ptr<std::atomic<bool>> Calling =
      std::make_shared<std::atomic<bool>>(false);
  std::shared_ptr<std::atomic<bool>> Done =
      std::make_shared<std::atomic<bool>>(false);
  std::shared_ptr<std::atomic<uint64_t>> Value =
      std::make_shared<std::atomic<uint64_t>>(0);
  std::shared_ptr<std::string> Error = std::make_shared<std::string>();
  std::thread T;

  QueryThread(sycl::event E, Query Q) {
    T = std::thread(
        [E, Q, Calling = Calling, Done = Done, Value = Value, Error = Error]() {
          Calling->store(true);
          try {
            Value->store(query(E, Q));
          } catch (const sycl::exception &Ex) {
            *Error = Ex.what();
          }
          Done->store(true);
        });
  }
  void join(const std::string &What) {
    if (!waitUntil([&] { return Done->load(); }, WaitTimeout)) {
      T.detach();
      fatal("query did not return: " + What);
    }
    T.join();
  }
};

enum class Kind { PerEventProfiling, QueueProfiling };

void runVariant(sycl::context &Ctx, sycl::device &Dev, Kind K, bool InOrder) {
  const std::string Variant =
      std::string(K == Kind::PerEventProfiling ? "per-event profiling"
                                               : "queue profiling") +
      (InOrder ? ", in-order queue" : ", out-of-order queue");
  std::cout << "Variant: " << Variant << std::endl;
  GateOpener Opener;

  sycl::property_list Props;
  if (K == Kind::QueueProfiling && InOrder)
    Props = sycl::property_list{sycl::property::queue::in_order{},
                                sycl::property::queue::enable_profiling{}};
  else if (K == Kind::QueueProfiling)
    Props = sycl::property_list{sycl::property::queue::enable_profiling{}};
  else if (InOrder)
    Props = sycl::property_list{sycl::property::queue::in_order{}};
  sycl::queue Q{Ctx, Dev, Props};

  sycl::event E =
      K == Kind::PerEventProfiling
          ? syclex::make_event(
                Ctx, syclex::properties{syclex::enable_profiling{true}})
          : syclex::make_event(Ctx);

  // Generation 1: an immediate signal, so that the event is a profiling tag
  // event with a backend event before the deferred generations.
  syclex::enqueue_signal_event(Q, E);
  boundedWait(E, Variant + ": generation 1");
  uint64_t PrevEnd =
      E.get_profiling_info<sycl::info::event_profiling::command_end>();
  CHECK(PrevEnd != 0);

  for (int Gen = 2; Gen <= Generations; ++Gen) {
    for (Query QK : {Query::End, Query::Start, Query::Submit}) {
      const std::string What =
          Variant + ": generation " + std::to_string(Gen) + ", " + name(QK);
      QueueGate.close();
      const int Base = QueueGate.Entered.load();
      Q.submit([&](sycl::handler &CGH) {
        CGH.host_task([]() { QueueGate.pass(); });
      });
      if (!waitUntil([&] { return QueueGate.Entered.load() == Base + 1; },
                     WaitTimeout))
        fatal(What + ": host task did not start");

      // Deferred signal: the in-order or barrier dependency on the gated host
      // task keeps it in the scheduler, without a backend event.
      syclex::enqueue_signal_event(Q, E);

      QueryThread QT(E, QK);
      if (!waitUntil([&] { return QT.Calling->load(); }, WaitTimeout))
        fatal(What + ": query thread did not start");
      // Best effort: pinned, the query blocks while the signal is deferred.
      std::this_thread::sleep_for(ObservationWindow);
      const bool ReturnedEarly = QT.Done->load();
      if (ReturnedEarly)
        std::cerr << What << ": returned " << QT.Value->load()
                  << " while the signal was deferred" << std::endl;
      CHECK(!ReturnedEarly);
      CHECK(!isComplete(E));

      QueueGate.open();
      QT.join(What);
      if (!QT.Error->empty()) {
        std::cerr << What << ": threw: " << *QT.Error << std::endl;
        ++Failures;
        continue;
      }
      CHECK(isComplete(E));

      // Deterministic once the gate is open: the new generation's tag
      // timestamp, not 0 and not the previous generation's value.
      using namespace sycl::info;
      const uint64_t Submit =
          E.get_profiling_info<event_profiling::command_submit>();
      const uint64_t Start =
          E.get_profiling_info<event_profiling::command_start>();
      const uint64_t End = E.get_profiling_info<event_profiling::command_end>();
      if (QT.Value->load() != End)
        std::cerr << What << ": returned " << QT.Value->load()
                  << ", command_end after completion is " << End << std::endl;
      CHECK(QT.Value->load() == End);
      CHECK(Submit == End);
      CHECK(Start == End);
      CHECK(End != 0);
      CHECK(End >= PrevEnd);
      PrevEnd = End;
    }
  }
  Q.wait_and_throw();
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
    for (Kind K : {Kind::PerEventProfiling, Kind::QueueProfiling}) {
      if (K == Kind::PerEventProfiling && !PerEvent)
        continue;
      for (bool InOrder : {true, false})
        runVariant(Ctx, Dev, K, InOrder);
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
