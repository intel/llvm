// REQUIRES: level_zero_v2_adapter && arch-intel_gpu_bmg_g21
// REQUIRES: aspect-usm_shared_allocations

// RUN: %{build} -o %t.out
// RUN: env SYCL_LAUNCH_BLOCKING=1 %{run} %t.out

// Submitting the finalized graph is a regular submission
// and does block: the test never waits on the queue, so the
// results are only there if blocking mode drained it.

#include "../../graph_common.hpp"

#include <sycl/properties/all_properties.hpp>

int main() {
  queue Queue{property::queue::in_order{}};

  exp_ext::command_graph Graph{
      Queue.get_context(),
      Queue.get_device(),
      {exp_ext::property::graph::enable_native_recording{}}};

  const size_t N = 1024;
  int *Data = malloc_shared<int>(N, Queue);
  std::fill(Data, Data + N, -1);

  Graph.begin_recording(Queue);
  Queue.submit([&](handler &CGH) {
    CGH.parallel_for(range<1>{N},
                     [=](id<1> Idx) { Data[Idx] = static_cast<int>(Idx); });
  });
  Graph.end_recording(Queue);

  auto ExecutableGraph = Graph.finalize();
  Queue.ext_oneapi_graph(ExecutableGraph);

  for (size_t I = 0; I < N; ++I)
    assert(check_value(I, static_cast<int>(I), Data[I], "Data"));

  sycl::free(Data, Queue);
  return 0;
}
