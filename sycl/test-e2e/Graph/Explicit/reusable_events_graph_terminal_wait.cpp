// REQUIRES: level_zero_v2_adapter

// RUN: %{build} -o %t.out
// RUN: env SYCL_QUEUE_THREAD_POOL_SIZE=8 %{run} %t.out

// tests-10-02 E18, control variants (guard): a direct wait on the event
// returned by ext_oneapi_graph, which has not been re-signaled, covers every
// terminal partition of the graph, including an independent host task.
//
// The queue-wait and re-signal variants of E18 are known defects
// (review-10-02 #4) and live in reusable_events_graph_queue_wait_terminals.cpp.
//
// Graph layout, in node insertion order:
//
//   HostA (gated, terminal)      HostB -> Kernel (terminal)
//
// The layout relies on the host-task partitioning at 8337da70
// (exec_graph_impl::makePartitions): HostA, HostB and Kernel get partitions
// in this order, so the returned event belongs to the Kernel partition
// (MPartitions.back()) and HostA's event is attached to it as an additional
// completion dependency.
//
// While HostA is gated, HostB and Kernel complete. A wait that returns before
// HostA completes reads MarkerA == 0 on return; this oracle is deterministic.
// The "wait is still blocked" check before opening the gate is best effort.

#include "../graph_common.hpp"

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
// Observation window of the best-effort "wait is still blocked" check.
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

// Runs Wait on a controlled thread while HostA is gated and checks that it
// does not return before HostA completes.
template <typename WaitFn> void observeWait(WaitFn Wait, const char *What) {
  auto Done = std::make_shared<std::atomic<bool>>(false);
  auto SawA = std::make_shared<std::atomic<int>>(-1);
  auto Error = std::make_shared<std::string>();
  std::thread Waiter([=]() mutable {
    try {
      Wait();
    } catch (const sycl::exception &Ex) {
      *Error = Ex.what();
    }
    SawA->store(MarkerA.load());
    Done->store(true);
  });

  // Best effort: a correct wait cannot return while HostA is gated.
  waitUntil([&] { return Done->load(); }, ObservationWindow);
  if (Done->load())
    std::cerr << What << ": returned while HostA was gated" << std::endl;
  CHECK(!Done->load());

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
  // Deterministic: the wait returned after HostA completed.
  CHECK(SawA->load() == 1);
}

void runControl(exp_ext::command_graph<exp_ext::graph_state::executable> &Exec,
                queue &Q, int *KernelOut, const char *What) {
  std::cout << "Control: " << What << std::endl;
  GateOpener Opener;
  Q.memset(KernelOut, 0, sizeof(int)).wait();
  MarkerA = 0;
  MarkerB = 0;
  const int Base = GateA.Entered.load();
  GateA.close();

  event E = Q.ext_oneapi_graph(Exec);
  if (!waitUntil(
          [&] { return GateA.Entered.load() == Base + 1 && MarkerB.load(); },
          WaitTimeout))
    fatal(std::string(What) + ": graph host tasks did not start");

  observeWait([E]() mutable { E.wait(); }, What);

  if (!waitUntil([&] { return GateA.Left.load() == Base + 1; }, WaitTimeout))
    fatal(std::string(What) + ": HostA did not finish");
  Q.wait_and_throw();
  int K = 0;
  Q.memcpy(&K, KernelOut, sizeof(int)).wait();
  CHECK(K == 1);
  CHECK(MarkerA.load() == 1);
  CHECK(MarkerB.load() == 1);
}

} // namespace

int main() {
  GateOpener Opener;
  try {
    device Dev;
    context Ctx{Dev};
    queue InOrderQ{Ctx, Dev, {property::queue::in_order{}}};
    queue OutOfOrderQ{Ctx, Dev};
    int *KernelOut = malloc_device<int>(1, InOrderQ);

    exp_ext::command_graph Graph{Ctx, Dev};
    buildGraph(Graph, KernelOut);
    auto Exec = Graph.finalize();

    runControl(Exec, InOrderQ, KernelOut, "event wait, in-order queue");
    runControl(Exec, OutOfOrderQ, KernelOut, "event wait, out-of-order queue");

    free(KernelOut, InOrderQ);
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
