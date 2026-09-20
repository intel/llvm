// REQUIRES: level_zero_v2_adapter && arch-intel_gpu_bmg_g21

// RUN: %{build} -o %t.out
// RUN: %{run} %t.out
// Extra run to check for leaks in Level Zero using UR_L0_LEAKS_DEBUG
// RUN: %if level_zero %{%{l0_leak_check} %{run} %t.out 2>&1 | FileCheck %s --implicit-check-not=LEAK %}

// Test that cross-queue dependencies recorded into a native recording graph
// survive the destruction of the events that expressed them. Two fork-joins are
// recorded, one using the events returned by submission and one using reusable
// events, and every event leaves scope before the graph is finalized. The graph
// is then replayed, so the recorded dependencies are all that remain to order
// the kernels.

#include "../../graph_common.hpp"

#include <sycl/ext/oneapi/experimental/reusable_events.hpp>
#include <sycl/properties/all_properties.hpp>

int main() {
  device Dev;
  context Ctx{Dev};

  queue Queue1{Ctx, Dev, {property::queue::in_order{}}};
  queue Queue2{Ctx, Dev, {property::queue::in_order{}}};

  QueueStateVerifier verifier(Queue1, Queue2);

  const size_t N = 1024;
  int *DataA = malloc_device<int>(N, Dev, Ctx);
  int *DataB = malloc_device<int>(N, Dev, Ctx);

  exp_ext::command_graph Graph{
      Ctx, Dev, {exp_ext::property::graph::enable_native_recording{}}};

  verifier.verify(EXECUTING, EXECUTING);

  Graph.begin_recording(Queue1);
  verifier.verify(RECORDING, EXECUTING);

  // First fork-join: dependencies expressed with the events returned by
  // submission. Both events are gone at the end of this scope.
  {
    event Fork = Queue1.parallel_for(
        range<1>{N}, [=](id<1> Idx) { DataA[Idx] = static_cast<int>(Idx); });

    event Branch = Queue2.submit([&](handler &CGH) {
      CGH.depends_on(Fork);
      CGH.parallel_for(range<1>{N}, [=](id<1> Idx) { DataA[Idx] *= 2; });
    });
    verifier.verify(RECORDING, RECORDING);

    Queue1.submit([&](handler &CGH) {
      CGH.depends_on(Branch);
      CGH.parallel_for(range<1>{N}, [=](id<1> Idx) { DataA[Idx] += 1; });
    });
  }

  // Second fork-join: the same shape, with a reusable event on each edge. Both
  // events are gone at the end of this scope.
  {
    event ForkEvent = exp_ext::make_event(Ctx);
    event JoinEvent = exp_ext::make_event(Ctx);

    Queue1.parallel_for(range<1>{N},
                        [=](id<1> Idx) { DataB[Idx] = static_cast<int>(Idx); });
    exp_ext::enqueue_signal_event(Queue1, ForkEvent);

    exp_ext::enqueue_wait_event(Queue2, ForkEvent);
    Queue2.parallel_for(range<1>{N}, [=](id<1> Idx) { DataB[Idx] *= 3; });
    exp_ext::enqueue_signal_event(Queue2, JoinEvent);

    exp_ext::enqueue_wait_event(Queue1, JoinEvent);
    Queue1.parallel_for(range<1>{N}, [=](id<1> Idx) { DataB[Idx] += 1; });
  }

  Graph.end_recording();
  verifier.verify(EXECUTING, EXECUTING);

  auto ExecutableGraph = Graph.finalize();

  // Each fork-join initializes its own output, so replaying is idempotent and a
  // second replay has to reproduce the same values.
  Queue1.ext_oneapi_graph(ExecutableGraph);
  Queue1.ext_oneapi_graph(ExecutableGraph).wait();

  std::vector<int> HostA(N), HostB(N);
  Queue1.memcpy(HostA.data(), DataA, N * sizeof(int));
  Queue1.memcpy(HostB.data(), DataB, N * sizeof(int));
  Queue1.wait();

  for (size_t i = 0; i < N; i++) {
    assert(check_value(i, static_cast<int>(i) * 2 + 1, HostA[i], "DataA"));
    assert(check_value(i, static_cast<int>(i) * 3 + 1, HostB[i], "DataB"));
  }

  free(DataA, Ctx);
  free(DataB, Ctx);

  return 0;
}
