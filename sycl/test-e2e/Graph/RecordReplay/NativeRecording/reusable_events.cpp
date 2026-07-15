// REQUIRES: level_zero_v2_adapter && arch-intel_gpu_bmg_g21
// REQUIRES: aspect-ext_oneapi_per_event_profiling

// RUN: %{build} -o %t.out
// RUN: %{run} %t.out
// Extra run to check for leaks in Level Zero using UR_L0_LEAKS_DEBUG
// RUN: %if level_zero %{%{l0_leak_check} %{run} %t.out 2>&1 | FileCheck %s --implicit-check-not=LEAK %}

// Test that a reusable event can be used eagerly outside a native recording
// graph, recorded to express cross-queue dependencies inside the graph, and
// then re-used eagerly again after graph replay. Exercised both with and
// without a profiling-enabled event.

#include "../../graph_common.hpp"

#include <sycl/ext/oneapi/experimental/reusable_events.hpp>
#include <sycl/properties/all_properties.hpp>

void run_test(bool ProfileEvent) {
  device Dev;
  context Ctx{Dev};

  queue Queue1{Ctx, Dev, {property::queue::in_order{}}};
  queue Queue2{Ctx, Dev, {property::queue::in_order{}}};

  QueueStateVerifier verifier(Queue1, Queue2);

  exp_ext::properties ProfProps{exp_ext::enable_profiling};
  event Ev = ProfileEvent ? exp_ext::make_event(Ctx, ProfProps)
                          : exp_ext::make_event(Ctx);

  const size_t N = 1024;
  int *Data = malloc_device<int>(N, Dev, Ctx);

  exp_ext::command_graph Graph{
      Ctx, Dev, {exp_ext::property::graph::enable_native_recording{}}};

  verifier.verify(EXECUTING, EXECUTING);

  // Use the event eagerly outside the graph.
  Queue2.parallel_for(range<1>{N},
                      [=](id<1> idx) { Data[idx] = static_cast<int>(idx); });
  exp_ext::enqueue_signal_event(Queue2, Ev);
  exp_ext::enqueue_wait_event(Queue1, Ev);

  // Record the event to signal dependencies inside the graph.
  Graph.begin_recording(Queue1);
  verifier.verify(RECORDING, EXECUTING);

  Queue1.parallel_for(range<1>{N}, [=](id<1> idx) { Data[idx] *= 2; });

  // Signal the event on Queue1 and wait on Queue2 to pull it into the graph.
  exp_ext::enqueue_signal_event(Queue1, Ev);
  exp_ext::enqueue_wait_event(Queue2, Ev);
  verifier.verify(RECORDING, RECORDING);

  Queue2.parallel_for(range<1>{N}, [=](id<1> idx) { Data[idx] += 1; });

  // Re-use the same event to signal Queue2's work and wait it back on
  // the main recording queue.
  exp_ext::enqueue_signal_event(Queue2, Ev);
  exp_ext::enqueue_wait_events(Queue1, {Ev});

  Graph.end_recording();
  verifier.verify(EXECUTING, EXECUTING);

  auto ExecutableGraph = Graph.finalize();

  // Replay the graph twice, then re-use the event eagerly again.
  Queue1.ext_oneapi_graph(ExecutableGraph);
  Queue1.ext_oneapi_graph(ExecutableGraph);
  exp_ext::enqueue_signal_event(Queue1, Ev);
  Ev.wait();

  if (ProfileEvent) {
    auto Start = Ev.get_profiling_info<info::event_profiling::command_start>();
    auto End = Ev.get_profiling_info<info::event_profiling::command_end>();
    assert(End >= Start);
  }

  std::vector<int> HostData(N);
  Queue1.memcpy(HostData.data(), Data, N * sizeof(int)).wait();

  for (size_t i = 0; i < N; i++) {
    int Expected = static_cast<int>(i) * 4 + 3;
    assert(check_value(i, Expected, HostData[i], "HostData"));
  }

  free(Data, Ctx);
}

int main() {
  run_test(/*ProfileEvent=*/false);
  run_test(/*ProfileEvent=*/true);
  return 0;
}
