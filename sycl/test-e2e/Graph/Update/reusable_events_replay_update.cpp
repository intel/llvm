// REQUIRES: level_zero_v2_adapter

// RUN: %{build} -o %t.out
// RUN: env SYCL_QUEUE_THREAD_POOL_SIZE=8 %{run} %t.out

// tests-10-02 E17 (guard): graph replay and update cannot overtake an earlier
// execution whose returned event was re-signaled on another queue.
//
// The executable graph is a gated host task followed by a kernel. The event
// returned by ext_oneapi_graph belongs to the kernel partition (the last
// partition), so it is a device event and can be re-signaled. Each round:
//
//   1. Execute the graph (first execution, blocked by the gate) and capture a
//      consumer of its returned event E on a third queue.
//   2. Re-signal E on another queue. NewFirst: the new signal completes while
//      the first execution is still blocked. OldFirst: the new signal is
//      deferred behind a host task on the signaling queue, so it completes
//      only after the graph work.
//   3. Request a second execution, then update the executable graph to new
//      parameters on a controlled thread (the update may block).
//   4. Open the gate and request a third execution.
//
// Checks: the re-signal does not complete or release the first execution;
// the second execution and the captured consumer still wait for the first
// execution; the first and second executions use the old parameters and the
// third execution uses the new ones; the update source graph can be destroyed
// while old work is pending. Rounds swap the parameters back and forth and run
// with an in-order and an out-of-order graph queue.

#include "../graph_common.hpp"

#include <sycl/atomic_ref.hpp>
#include <sycl/ext/oneapi/experimental/reusable_events.hpp>

#include <atomic>
#include <chrono>
#include <cstdlib>
#include <memory>
#include <string>
#include <thread>
#include <vector>

namespace {

using namespace std::chrono_literals;

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

// A host-side gate passed by host tasks. A closed gate blocks pass() until it
// is opened, or until GateTimeout expires, in which case TimedOut is set so
// that a broken test fails instead of hanging.
struct Gate {
  std::mutex M;
  std::condition_variable CV;
  bool IsOpen = true;
  std::atomic<int> Entered{0};
  std::atomic<int> Left{0};
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
    {
      std::unique_lock<std::mutex> L(M);
      if (!CV.wait_for(L, GateTimeout, [this] { return IsOpen; }))
        TimedOut = true;
    }
    ++Left;
  }
};

Gate GraphGate;  // Passed by the graph's host task node.
Gate SignalGate; // Blocks the signaling queue in the OldFirst variant.

void openAllGates() {
  GraphGate.open();
  SignalGate.open();
}

// Opens every gate when it goes out of scope, so that no exit path leaves a
// host task blocked.
struct GateOpener {
  ~GateOpener() { openAllGates(); }
};

[[noreturn]] void fatal(const std::string &Msg) {
  std::cerr << "FATAL: " << Msg << std::endl;
  openAllGates();
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

bool isComplete(const event &E) {
  return E.get_info<info::event::command_execution_status>() ==
         info::event_command_status::complete;
}

// event::wait with a deadline: the wait runs on a helper thread, and a wait
// which does not return within WaitTimeout terminates the test.
void boundedWait(event E, const std::string &What) {
  auto Done = std::make_shared<std::atomic<bool>>(false);
  auto Error = std::make_shared<std::string>();
  std::thread Waiter([E, Done, Error]() mutable {
    try {
      E.wait();
    } catch (const sycl::exception &Ex) {
      *Error = Ex.what();
    }
    Done->store(true);
  });
  if (!waitUntil([&] { return Done->load(); }, WaitTimeout)) {
    Waiter.detach();
    fatal("wait timed out: " + What);
  }
  Waiter.join();
  if (!Error->empty()) {
    std::cerr << "wait failed: " << What << ": " << *Error << std::endl;
    ++Failures;
  }
}

using Atomic = sycl::atomic_ref<int, sycl::memory_order::relaxed,
                                sycl::memory_scope::device,
                                sycl::access::address_space::global_space>;

// Both the initial graph and every update source are built by this function,
// so that their nodes have identical types and topology, as required by whole
// graph update. Only the kernel's output pointer differs.
void addNodes(exp_ext::command_graph<exp_ext::graph_state::modifiable> &Graph,
              int *Out) {
  auto HostTask = Graph.add(
      [&](handler &CGH) { CGH.host_task([]() { GraphGate.pass(); }); });
  Graph.add(
      [&](handler &CGH) {
        CGH.single_task([=]() { Atomic(*Out).fetch_add(1); });
      },
      {exp_ext::property::node::depends_on(HostTask)});
}

int readDevice(queue &Q, int *Ptr) {
  int Value = 0;
  Q.memcpy(&Value, Ptr, sizeof(int)).wait();
  return Value;
}

enum class Ordering { NewFirst, OldFirst };

const char *name(Ordering O) {
  return O == Ordering::NewFirst ? "NewFirst" : "OldFirst";
}

void runVariant(context &Ctx, device &Dev, bool InOrderGraphQueue,
                Ordering Order) {
  std::cout << "Variant: " << (InOrderGraphQueue ? "in-order" : "out-of-order")
            << " graph queue, " << name(Order) << std::endl;

  property_list GraphQueueProps;
  if (InOrderGraphQueue)
    GraphQueueProps = property_list{property::queue::in_order{}};
  queue Q{Ctx, Dev, GraphQueueProps};
  queue Q2{Ctx, Dev, {property::queue::in_order{}}}; // Re-signals E.
  queue Q3{Ctx, Dev};                                // Captured consumer.

  int *Ptrs[2] = {malloc_device<int>(1, Q), malloc_device<int>(1, Q)};
  int *Obs = malloc_device<int>(1, Q);
  Q.memset(Ptrs[0], 0, sizeof(int));
  Q.memset(Ptrs[1], 0, sizeof(int));
  Q.memset(Obs, 0, sizeof(int));
  Q.wait_and_throw();
  int Expected[2] = {0, 0};

  exp_ext::command_graph Graph{Ctx, Dev};
  addNodes(Graph, Ptrs[0]);
  auto Exec = Graph.finalize(exp_ext::property::graph::updatable{});
  int Current = 0; // Index of the pointer the executable graph writes.

  for (int Round = 0; Round < 2; ++Round) {
    GateOpener Opener;
    const int Old = Current;
    const int New = 1 - Current;
    const int Base = GraphGate.Entered.load();
    CHECK(GraphGate.Left.load() == Base);
    GraphGate.close();

    // 1. First execution; its host task blocks on the gate.
    event E = Q.ext_oneapi_graph(Exec);
    if (!waitUntil([&] { return GraphGate.Entered.load() == Base + 1; },
                   WaitTimeout))
      fatal("first execution's host task did not start");

    // Consumer of the first execution's signal, captured before re-signaling.
    event C1 = Q3.submit([&](handler &CGH) {
      CGH.depends_on(E);
      int *OldPtr = Ptrs[Old];
      CGH.single_task([=]() { *Obs = Atomic(*OldPtr).load(); });
    });

    // 2. Re-signal E on another queue.
    if (Order == Ordering::OldFirst) {
      SignalGate.close();
      const int SignalBase = SignalGate.Entered.load();
      Q2.submit(
          [&](handler &CGH) { CGH.host_task([]() { SignalGate.pass(); }); });
      if (!waitUntil(
              [&] { return SignalGate.Entered.load() == SignalBase + 1; },
              WaitTimeout))
        fatal("signal queue's host task did not start");
      // Deferred: Q2 is blocked by its host task.
      exp_ext::enqueue_signal_event(Q2, E);
    } else {
      exp_ext::enqueue_signal_event(Q2, E);
      // The new signal completes although the first execution is blocked.
      boundedWait(E, "new signal of E");
      CHECK(GraphGate.Left.load() == Base);
    }

    // 3. Second execution, requested after the re-signal.
    event E2 = Q.ext_oneapi_graph(Exec);

    // Best effort: these cannot complete while the gate is closed; a bounded
    // sleep gives a premature completion the chance to show.
    std::this_thread::sleep_for(PendingWindow);
    CHECK(GraphGate.Entered.load() == Base + 1);
    CHECK(!isComplete(C1));
    CHECK(!isComplete(E2));
    if (Order == Ordering::OldFirst)
      CHECK(!isComplete(E));

    // Update on a controlled thread. The update may block until the previous
    // executions complete. The update source is destroyed on that thread as
    // soon as the update returns, possibly before the old work finishes.
    std::atomic<bool> UpdateDone{false};
    std::string UpdateError;
    std::thread Updater([&]() {
      try {
        exp_ext::command_graph Source{Ctx, Dev};
        addNodes(Source, Ptrs[New]);
        Exec.update(Source);
      } catch (const sycl::exception &Ex) {
        UpdateError = Ex.what();
      }
      UpdateDone = true;
    });
    std::this_thread::sleep_for(PendingWindow);
    // Informational only: the specification allows update to return early.
    std::cout << "  round " << Round << ": update "
              << (UpdateDone.load() ? "returned" : "still blocked")
              << " while the first execution was gated" << std::endl;

    // 4. Release the first execution.
    GraphGate.open();
    if (!waitUntil([&] { return UpdateDone.load(); }, WaitTimeout)) {
      Updater.detach();
      fatal("graph update did not return");
    }
    Updater.join();
    if (!UpdateError.empty()) {
      std::cerr << "update failed: " << UpdateError << std::endl;
      ++Failures;
    }

    // Third execution, with the new parameters.
    event E3 = Q.ext_oneapi_graph(Exec);

    boundedWait(E2, "second execution");
    boundedWait(C1, "captured consumer");
    boundedWait(E3, "third execution");

    if (Order == Ordering::OldFirst) {
      // The deferred signal of E stays behind Q2's host task, which is still
      // blocked; only the graph work has completed.
      CHECK(!isComplete(E));
      SignalGate.open();
      boundedWait(E, "deferred new signal of E");
    }
    Q.wait_and_throw();
    Q2.wait_and_throw();
    Q3.wait_and_throw();

    Expected[Old] += 2; // First and second execution.
    Expected[New] += 1; // Third execution.
    CHECK(readDevice(Q, Ptrs[Old]) == Expected[Old]);
    CHECK(readDevice(Q, Ptrs[New]) == Expected[New]);
    // The consumer ran after the first execution; the second execution may or
    // may not have finished by then.
    const int Observed = readDevice(Q, Obs);
    CHECK(Observed >= Expected[Old] - 1 && Observed <= Expected[Old]);
    CHECK(GraphGate.Entered.load() == Base + 3);
    CHECK(GraphGate.Left.load() == Base + 3);

    Current = New;
  }

  free(Ptrs[0], Q);
  free(Ptrs[1], Q);
  free(Obs, Q);
}

} // namespace

int main() {
  GateOpener Opener;
  try {
    device Dev;
    context Ctx{Dev};
    for (bool InOrder : {false, true})
      for (Ordering Order : {Ordering::NewFirst, Ordering::OldFirst})
        runVariant(Ctx, Dev, InOrder, Order);
  } catch (const sycl::exception &Ex) {
    std::cerr << "Unexpected exception: " << Ex.what() << std::endl;
    return 1;
  }
  CHECK(!GraphGate.TimedOut.load());
  CHECK(!SignalGate.TimedOut.load());
  if (Failures) {
    std::cerr << Failures << " check(s) failed" << std::endl;
    return 1;
  }
  return 0;
}
