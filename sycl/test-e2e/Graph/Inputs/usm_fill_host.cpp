// Tests adding a host USM fill operation as a graph node.

#include "../graph_common.hpp"

int main() {
  queue Queue{};

  if (!are_graphs_supported(Queue)) {
    return 0;
  }

  if (!Queue.get_device().has(sycl::aspect::usm_host_allocations)) {
    return 0;
  }

  exp_ext::command_graph Graph{Queue.get_context(), Queue.get_device()};

  const size_t N = 10;
  int *Arr = malloc_host<int>(N, Queue);

  int Pattern = 3;
  auto NodeA =
      add_node(Graph, Queue, [&](handler &CGH) { CGH.fill(Arr, Pattern, N); });

  auto ExecGraph = Graph.finalize();

  Queue.submit([&](handler &CGH) { CGH.ext_oneapi_graph(ExecGraph); }).wait();

  for (size_t i = 0; i < N; i++)
    assert(check_value(i, Pattern, Arr[i], "Arr"));

  sycl::free(Arr, Queue);

  return 0;
}
