// REQUIRES: level_zero_v2_adapter && arch-intel_gpu_bmg_g21

// RUN: %{build} -o %t.out
// RUN: %{run} %t.out
// RUN: env SYCL_LAUNCH_BLOCKING=1 %{run} %t.out

// Native recording does not set the queue's command graph, so
// SYCL_LAUNCH_BLOCKING must recognise it separately - otherwise recording a
// command would wait on an event that only signals when the finalized graph is
// submitted, and hang.

#include "../../graph_common.hpp"

#include <sycl/properties/all_properties.hpp>

int main() {
  queue Queue{property::queue::in_order{}};

  exp_ext::command_graph Graph{
      Queue.get_context(),
      Queue.get_device(),
      {exp_ext::property::graph::enable_native_recording{}}};

  const size_t N = 1024;
  int *Data = malloc_device<int>(N, Queue);

  Graph.begin_recording(Queue);
  Queue.submit([&](handler &CGH) {
    CGH.parallel_for(range<1>{N},
                     [=](id<1> Idx) { Data[Idx] = static_cast<int>(Idx); });
  });
  Graph.end_recording(Queue);

  auto ExecutableGraph = Graph.finalize();
  Queue.ext_oneapi_graph(ExecutableGraph);
  Queue.wait_and_throw();

  std::vector<int> Host(N, -1);
  Queue.memcpy(Host.data(), Data, N * sizeof(int)).wait();
  for (size_t I = 0; I < N; ++I)
    assert(Host[I] == static_cast<int>(I));

  sycl::free(Data, Queue);
  return 0;
}
