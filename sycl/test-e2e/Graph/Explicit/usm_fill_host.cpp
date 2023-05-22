// REQUIRES: level_zero, gpu
// RUN: %{build} -o %t.out
// RUN: %{run} %t.out

// Tests adding a host USM fill operation as a graph node with the explicit
// API.

#include "../graph_common.hpp"

int main() {

  queue Queue{gpu_selector_v};
  if (!Queue.get_device().has(sycl::aspect::usm_host_allocations)) {
    return 0;
  }

  exp_ext::command_graph Graph{Queue.get_context(), Queue.get_device()};

  const size_t N = 10;
  float *Arr = malloc_host<float>(N, Queue);

  float Pattern = 3.14f;
  auto NodeA = Graph.add([&](handler &CGH) { CGH.fill(Arr, Pattern, N); });

  auto ExecGraph = Graph.finalize();

  Queue.submit([&](handler &CGH) { CGH.ext_oneapi_graph(ExecGraph); }).wait();

  for (int i = 0; i < N; i++)
    assert(Arr[i] == Pattern);

  sycl::free(Arr, Queue);

  return 0;
}
