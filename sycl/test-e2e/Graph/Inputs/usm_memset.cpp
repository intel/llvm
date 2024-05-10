// Tests adding a USM memset operation as a graph node.

#include "../graph_common.hpp"

int main() {

  queue Queue;

  exp_ext::command_graph Graph{Queue.get_context(), Queue.get_device()};

  const size_t N = 10;
  unsigned char *Arr = malloc_device<unsigned char>(N, Queue);

  int Value = 77;
  auto NodeA =
      add_node(Graph, Queue, [&](handler &CGH) { CGH.memset(Arr, Value, N); });

  auto ExecGraph = Graph.finalize();

  Queue.submit([&](handler &CGH) { CGH.ext_oneapi_graph(ExecGraph); }).wait();

  std::vector<unsigned char> Output(N);
  Queue.memcpy(Output.data(), Arr, N).wait();
  for (int i = 0; i < N; i++)
    assert(Output[i] == Value);

  sycl::free(Arr, Queue);

  return 0;
}
