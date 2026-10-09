// REQUIRES: level_zero_v2_adapter

// RUN: %{build} -o %t.out
// RUN: env SYCL_QUEUE_THREAD_POOL_SIZE=8 %{run} %t.out

// XFAIL: *
// XFAIL-TRACKER: review-10-02 #4 (queue waits ignore graph terminal partitions
// attached via MPostCompleteEvents)

// tests-10-02 E18 (known defect): queue waits and waits on an original
// consumer of a graph execution include every terminal partition of the
// graph, including an independent host task, also after the returned event
// has been re-signaled.
//
// At 8337da70 queue::wait uses the binding captured at submission:
// waitForDependency(LastEvent) on an in-order queue and
// Scheduler::waitForEvent(binding) over MEventsWeak on an out-of-order queue.
// event_binding::wait ignores the event_impl's MPostCompleteEvents, which is
// where the executable graph attaches its other terminal partitions, and
// urQueueFinish does not cover host tasks. The waits therefore return while
// HostA is still gated. The direct event wait control variants are in
// reusable_events_graph_terminal_wait.cpp and stay enabled.
//
// The consumer variant (a command submitted with depends_on(E) before E is
// re-signaled) also fails at 8337da70 independently of re-signaling: a
// dependency captures only the binding of the returned partition and never
// the attached terminal partitions.
//
// Graph layout, in node insertion order:
//
//   HostA (gated, terminal)      HostB -> Kernel (terminal)
//
// The layout relies on the host-task partitioning at 8337da70
// (exec_graph_impl::makePartitions): the returned event belongs to the Kernel
// partition (MPartitions.back()), so it is a device event which can be
// re-signaled, and HostA's event is attached to it. If the layout changes so
// that the returned event is a host event, enqueue_signal_event throws
// errc::invalid and the variant reports a broken layout assumption.
//
// Determinism: a wait or consumer that runs before HostA completes reads
// MarkerA == 0; that oracle is deterministic. The test only fails at the head
// if the premature return happens within ObservationWindow after HostB
// completes, which holds unless the kernel partition takes longer than that.
// E.wait() is never called before the gate opens in the re-signal variants,
// and E is never re-signaled while a thread waits on it.

#include "../graph_common.hpp"

#include <sycl/ext/oneapi/experimental/reusable_events.hpp>

#include <atomic>
#include <chrono>
#include <cstdlib>
#include <memory>
#include <string>
#include <thread>

namespace {

using namespace std::chrono_literals;

constexpr auto GateTimeout = 60s;
constexpr auto WaitTimeout = 60s;
// How long a wait has to return prematurely while HostA is gated.
constexpr auto ObservationWindow = 2s;

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

Gate GateA;
std::atomic<int> MarkerA{0};
std::atomic<int> MarkerB{0};

struct GateOpener {
  ~GateOpener() { GateA.open(); }
};

[[noreturn]] void fatal(const std::string &Msg) {
  std::cerr << "FATAL: " << Msg << std::endl;
  GateA.open();
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

void buildGraph(exp_ext::command_graph<exp_ext::graph_state::modifiable> &Graph,
                int *KernelOut) {
  // Independent nodes, no buffers: no edge other than HostB -> Kernel.
  Graph.add([&](handler &CGH) {
    CGH.host_task([]() {
      GateA.pass();
      MarkerA = 1;
    });
  });
  auto HostB =
      Graph.add([&](handler &CGH) { CGH.host_task([]() { MarkerB = 1; }); });
  Graph.add([&](handler &CGH) { CGH.single_task([=]() { *KernelOut = 1; }); },
            {exp_ext::property::node::depends_on(HostB)});
}

// Runs Wait on a controlled thread while HostA is gated. Wait returns the
// value of MarkerA observed by whatever completed (the waiting thread itself,
// or a consumer).
template <typename WaitFn> void observeWait(WaitFn Wait, const char *What) {
  auto Done = std::make_shared<std::atomic<bool>>(false);
  auto SawA = std::make_shared<std::atomic<int>>(-1);
  auto Error = std::make_shared<std::string>();
  std::thread Waiter([=]() mutable {
    int Seen = -1;
    try {
      Seen = Wait();
    } catch (const sycl::exception &Ex) {
      *Error = Ex.what();
    }
    SawA->store(Seen);
    Done->store(true);
  });

  waitUntil([&] { return Done->load(); }, ObservationWindow);
  const bool ReturnedEarly = Done->load();
  if (ReturnedEarly)
    std::cerr << What << ": returned while HostA was gated (MarkerA "
              << SawA->load() << ")" << std::endl;
  CHECK(!ReturnedEarly);

  GateA.open();
  if (!waitUntil([&] { return Done->load(); }, WaitTimeout)) {
    Waiter.detach();
    fatal(std::string(What) + ": wait did not return");
  }
  Waiter.join();
  if (!Error->empty()) {
    std::cerr << What << ": " << *Error << std::endl;
    ++Failures;
  }
  // Deterministic: whatever returned did so after HostA completed.
  CHECK(SawA->load() == 1);
}

enum class Kind { QueueWait, ResignalQueueWait, ResignalConsumer };

const char *name(Kind K) {
  switch (K) {
  case Kind::QueueWait:
    return "queue wait";
  case Kind::ResignalQueueWait:
    return "re-signal, then queue wait";
  case Kind::ResignalConsumer:
    return "re-signal, then wait on an original consumer";
  }
  return "";
}

void runVariant(exp_ext::command_graph<exp_ext::graph_state::executable> &Exec,
                context &Ctx, device &Dev, bool InOrder, Kind K,
                int *KernelOut) {
  std::string What = std::string(name(K)) +
                     (InOrder ? ", in-order queue" : ", out-of-order queue");
  std::cout << "Variant: " << What << std::endl;

  property_list Props;
  if (InOrder)
    Props = property_list{property::queue::in_order{}};
  queue Q{Ctx, Dev, Props};
  queue SignalQ{Ctx, Dev, {property::queue::in_order{}}};
  queue ConsumerQ{Ctx, Dev};

  GateOpener Opener;
  Q.memset(KernelOut, 0, sizeof(int)).wait();
  MarkerA = 0;
  MarkerB = 0;
  const int Base = GateA.Entered.load();
  GateA.close();

  event E = Q.ext_oneapi_graph(Exec);

  // Original consumer, captured before E is re-signaled. It records MarkerA
  // when it runs.
  auto ConsumerSawA = std::make_shared<std::atomic<int>>(-1);
  event Consumer;
  if (K == Kind::ResignalConsumer)
    Consumer = ConsumerQ.submit([&](handler &CGH) {
      CGH.depends_on(E);
      CGH.host_task([ConsumerSawA]() { ConsumerSawA->store(MarkerA.load()); });
    });

  if (!waitUntil(
          [&] { return GateA.Entered.load() == Base + 1 && MarkerB.load(); },
          WaitTimeout))
    fatal(What + ": graph host tasks did not start");

  if (K != Kind::QueueWait) {
    try {
      exp_ext::enqueue_signal_event(SignalQ, E);
    } catch (const sycl::exception &Ex) {
      std::cerr << What << ": cannot re-signal the returned event ("
                << Ex.what() << "); the graph layout assumption no longer holds"
                << std::endl;
      ++Failures;
    }
    // Completes the new signal without waiting on E itself.
    SignalQ.wait_and_throw();
  }

  if (K == Kind::ResignalConsumer)
    observeWait(
        [Consumer, ConsumerSawA]() mutable {
          Consumer.wait();
          return ConsumerSawA->load();
        },
        What.c_str());
  else
    observeWait(
        [&Q]() {
          Q.wait();
          return MarkerA.load();
        },
        What.c_str());

  // Release everything: the gate is open now.
  if (!waitUntil([&] { return GateA.Left.load() == Base + 1; }, WaitTimeout))
    fatal(What + ": HostA did not finish");
  E.wait();
  Q.wait_and_throw();
  ConsumerQ.wait_and_throw();
  int Kernel = 0;
  Q.memcpy(&Kernel, KernelOut, sizeof(int)).wait();
  CHECK(Kernel == 1);
  CHECK(MarkerA.load() == 1);
  CHECK(MarkerB.load() == 1);
}

} // namespace

int main() {
  GateOpener Opener;
  try {
    device Dev;
    context Ctx{Dev};
    queue AllocQ{Ctx, Dev};
    int *KernelOut = malloc_device<int>(1, AllocQ);

    exp_ext::command_graph Graph{Ctx, Dev};
    buildGraph(Graph, KernelOut);
    auto Exec = Graph.finalize();

    for (Kind K :
         {Kind::QueueWait, Kind::ResignalQueueWait, Kind::ResignalConsumer})
      for (bool InOrder : {true, false})
        runVariant(Exec, Ctx, Dev, InOrder, K, KernelOut);

    free(KernelOut, AllocQ);
  } catch (const sycl::exception &Ex) {
    std::cerr << "Unexpected exception: " << Ex.what() << std::endl;
    return 1;
  }
  CHECK(!GateA.TimedOut.load());
  if (Failures) {
    std::cerr << Failures << " check(s) failed" << std::endl;
    return 1;
  }
  return 0;
}
