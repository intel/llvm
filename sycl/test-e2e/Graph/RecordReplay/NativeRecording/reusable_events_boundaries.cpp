// REQUIRES: level_zero_v2_adapter && arch-intel_gpu_bmg_g21

// RUN: %{build} -o %t.out
// RUN: env SYCL_QUEUE_THREAD_POOL_SIZE=8 %{run} %t.out
// Extra run to check for leaks in Level Zero using UR_L0_LEAKS_DEBUG
// RUN: %if level_zero %{%{l0_leak_check} env SYCL_QUEUE_THREAD_POOL_SIZE=8 %{run} %t.out 2>&1 | FileCheck %s --implicit-check-not=LEAK %}

// tests-10-02 E19, native recording variants (guard): reusable events at the
// boundaries of a native recording graph, signaled repeatedly.
//
// Every round:
//   1. External work writes a new input on the external queue and signals
//      EvIn; the recording queue waits for EvIn. Round 0 does this before
//      begin_recording, the other rounds before their replay.
//   2. EvIn is re-signaled before the replay completes (Separate events only,
//      see below).
//   3. The graph is replayed: Out = Input * 2 + 1, with an internal fork-join
//      through a reusable event (cross-queue) or a signal and wait on the
//      recording queue itself (same-queue).
//   4. The recording queue signals EvOut after the replay; the consumer queue
//      waits for it and writes Result[Round] = Out + Bias. EvOut is then
//      re-signaled before the consumer completes.
//
// Check: every replay observes its round's external input and every consumer
// observes its round's replay, although the public events have been
// re-signaled since the dependencies were captured, and their latest queue or
// recording state differs from the captured one.
//
// Events: Separate uses EvIn, EvOut and a graph-internal event, so they can be
// re-signaled on an unrelated queue at any time. Shared uses one event for all
// three; events signaled during native capture may not be used outside the
// graph, so Shared only signals the event eagerly when no replay is in flight
// or on the recording queue after the replay (as reusable_events.cpp does),
// never on another queue while a replay is in flight.
//
// Only scheduler-bypass (immediate) signals and waits are used: native
// recording forbids handler::host_task, buffers and out-of-order queues.

#include "../../graph_common.hpp"

#include <sycl/ext/oneapi/experimental/reusable_events.hpp>
#include <sycl/properties/all_properties.hpp>

#include <string>

namespace {

constexpr size_t N = 1024;
constexpr int Rounds = 3;
constexpr int Bias = 7;

int Failures = 0;

int inputValue(int Round, size_t I) {
  return Round * 1000 + static_cast<int>(I);
}

void runTest(bool CrossQueue, bool SharedEvent) {
  std::cout << "Variant: " << (CrossQueue ? "cross-queue" : "same-queue")
            << ", " << (SharedEvent ? "shared event" : "separate events")
            << std::endl;

  device Dev;
  context Ctx{Dev};
  const property_list InOrder{property::queue::in_order{}};

  queue Queue1{Ctx, Dev, InOrder}; // Recording queue.
  queue Queue2{Ctx, Dev, InOrder}; // Forked recording queue (cross-queue).
  queue ExtQueue{Ctx, Dev, InOrder};
  queue ConsQueue{Ctx, Dev, InOrder};
  queue IdleQueue{Ctx, Dev, InOrder}; // Re-signals, separate events only.

  // In the same-queue variant, every boundary uses the recording queue.
  queue &Ext = CrossQueue ? ExtQueue : Queue1;
  queue &Cons = CrossQueue ? ConsQueue : Queue1;

  QueueStateVerifier verifier(Queue1, Queue2);

  event Shared = exp_ext::make_event(Ctx);
  event EvIn = SharedEvent ? Shared : exp_ext::make_event(Ctx);
  event EvOut = SharedEvent ? Shared : exp_ext::make_event(Ctx);
  event EvGraph = SharedEvent ? Shared : exp_ext::make_event(Ctx);

  int *Input = malloc_device<int>(N, Dev, Ctx);
  int *Out = malloc_device<int>(N, Dev, Ctx);
  int *Result = malloc_device<int>(N * Rounds, Dev, Ctx);
  Queue1.memset(Out, 0, N * sizeof(int)).wait();
  Queue1.memset(Result, 0, N * Rounds * sizeof(int)).wait();

  // Re-signals E before work captured from its previous signal completes.
  auto Resignal = [&](event &E) {
    if (SharedEvent) {
      // Ordered after the replay on the recording queue.
      exp_ext::enqueue_signal_event(Queue1, E);
    } else {
      exp_ext::enqueue_signal_event(IdleQueue, E);
    }
  };

  auto ExternalWork = [&](int Round) {
    Ext.parallel_for(range<1>{N}, [=](id<1> Idx) {
      Input[Idx] = Round * 1000 + static_cast<int>(Idx[0]);
    });
    exp_ext::enqueue_signal_event(Ext, EvIn);
    exp_ext::enqueue_wait_event(Queue1, EvIn);
    if (!SharedEvent)
      Resignal(EvIn);
  };

  exp_ext::command_graph Graph{
      Ctx, Dev, {exp_ext::property::graph::enable_native_recording{}}};

  // Round 0's external work is captured before recording begins.
  ExternalWork(0);

  Graph.begin_recording(Queue1);
  verifier.verify(RECORDING, EXECUTING);
  Queue1.parallel_for(range<1>{N},
                      [=](id<1> Idx) { Out[Idx] = Input[Idx] * 2; });
  if (CrossQueue) {
    exp_ext::enqueue_signal_event(Queue1, EvGraph);
    exp_ext::enqueue_wait_event(Queue2, EvGraph);
    verifier.verify(RECORDING, RECORDING);
    Queue2.parallel_for(range<1>{N}, [=](id<1> Idx) { Out[Idx] += 1; });
    exp_ext::enqueue_signal_event(Queue2, EvGraph);
    exp_ext::enqueue_wait_events(Queue1, {EvGraph});
  } else {
    exp_ext::enqueue_signal_event(Queue1, EvGraph);
    exp_ext::enqueue_wait_event(Queue1, EvGraph);
    Queue1.parallel_for(range<1>{N}, [=](id<1> Idx) { Out[Idx] += 1; });
  }
  Graph.end_recording();
  verifier.verify(EXECUTING, EXECUTING);

  auto Exec = Graph.finalize();

  for (int Round = 0; Round < Rounds; ++Round) {
    if (Round > 0)
      ExternalWork(Round);

    Queue1.ext_oneapi_graph(Exec);

    exp_ext::enqueue_signal_event(Queue1, EvOut);
    exp_ext::enqueue_wait_event(Cons, EvOut);
    int *Dst = Result + Round * N;
    Cons.parallel_for(range<1>{N},
                      [=](id<1> Idx) { Dst[Idx] = Out[Idx] + Bias; });
    Resignal(EvOut);

    // Nothing is in flight before the next round's eager signals.
    Cons.wait_and_throw();
    Queue1.wait_and_throw();
    Queue2.wait_and_throw();
    Ext.wait_and_throw();
    IdleQueue.wait_and_throw();
  }

  // The events stay usable eagerly after the last replay.
  exp_ext::enqueue_signal_event(Queue1, Shared);
  Shared.wait();

  std::vector<int> Host(N * Rounds);
  Queue1.memcpy(Host.data(), Result, N * Rounds * sizeof(int)).wait();
  for (int Round = 0; Round < Rounds; ++Round) {
    for (size_t I = 0; I < N; ++I) {
      const int Expected = inputValue(Round, I) * 2 + 1 + Bias;
      if (!check_value(I, Expected, Host[Round * N + I],
                       ("Result[" + std::to_string(Round) + "]").c_str())) {
        ++Failures;
        break;
      }
    }
  }

  free(Input, Ctx);
  free(Out, Ctx);
  free(Result, Ctx);
}

} // namespace

int main() {
  for (bool CrossQueue : {false, true})
    for (bool SharedEvent : {false, true})
      runTest(CrossQueue, SharedEvent);
  if (Failures) {
    std::cerr << Failures << " check(s) failed" << std::endl;
    return 1;
  }
  return 0;
}
