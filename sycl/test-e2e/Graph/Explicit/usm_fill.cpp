// REQUIRES: level_zero, gpu
// RUN: %{build} -o %t.out
// RUN: %{run} %t.out

// Tests adding a USM fill operation as a graph node with the explicit API.

#include "../graph_common.hpp"

int main() {

  queue Queue{gpu_selector_v};

  exp_ext::command_graph Graph{Queue.get_context(), Queue.get_device()};

  const size_t N = 10;
  float *Arr = malloc_device<float>(N, Queue);

  float Pattern = 3.14f;
  auto NodeA = Graph.add([&](handler &CGH) { CGH.fill(Arr, Pattern, N); });

  auto ExecGraph = Graph.finalize();

  Queue.submit([&](handler &CGH) { CGH.ext_oneapi_graph(ExecGraph); }).wait();

  std::vector<float> Output(N);
  Queue.memcpy(Output.data(), Arr, N * sizeof(float)).wait();
  for (int i = 0; i < N; i++)
    assert(Output[i] == Pattern);

  sycl::free(Arr, Queue);

  return 0;
}
