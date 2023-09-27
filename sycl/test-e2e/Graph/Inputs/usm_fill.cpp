// Tests adding a USM fill operation as a graph node.

#include "../graph_common.hpp"

int main() {

  queue Queue{{sycl::ext::intel::property::queue::no_immediate_command_list{}}};

  exp_ext::command_graph Graph{Queue.get_context(), Queue.get_device()};

  const size_t N = 10;
  int *Arr = malloc_device<int>(N, Queue);

  int Pattern = 314;
  auto NodeA =
      add_node(Graph, Queue, [&](handler &CGH) { CGH.fill(Arr, Pattern, N); });

  auto ExecGraph = Graph.finalize();

  Queue.submit([&](handler &CGH) { CGH.ext_oneapi_graph(ExecGraph); }).wait();

  std::vector<int> Output(N);
  Queue.memcpy(Output.data(), Arr, N * sizeof(int)).wait();
  for (int i = 0; i < N; i++)
    assert(Output[i] == Pattern);

  sycl::free(Arr, Queue);

  return 0;
}
