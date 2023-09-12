// Tests creating a parent graph with multiple submissions of the same subgraph
// in it.

#include "../graph_common.hpp"

int main() {
  queue Queue;

  exp_ext::command_graph Graph{Queue.get_context(), Queue.get_device()};
  exp_ext::command_graph SubGraph{Queue.get_context(), Queue.get_device()};

  const size_t N = 10;
  float *X = malloc_device<float>(N, Queue);

  auto S1 = add_node(SubGraph, Queue, [&](handler &CGH) {
    CGH.parallel_for(N, [=](id<1> it) { X[it] *= 2.0f; });
  });

  add_node(
      SubGraph, Queue,
      [&](handler &CGH) {
        depends_on_helper(CGH, S1);
        CGH.parallel_for(N, [=](id<1> it) { X[it] += 0.5f; });
      },
      S1);

  auto ExecSubGraph = SubGraph.finalize();

  auto P1 = add_node(Graph, Queue, [&](handler &CGH) {
    CGH.parallel_for(N, [=](id<1> it) { X[it] = 1.0f; });
  });

  auto P2 = add_node(
      Graph, Queue,
      [&](handler &CGH) {
        depends_on_helper(CGH, P1);
        CGH.ext_oneapi_graph(ExecSubGraph);
      },
      P1);

  auto P3 = add_node(
      Graph, Queue,
      [&](handler &CGH) {
        depends_on_helper(CGH, P2);
        CGH.parallel_for(range<1>{N}, [=](id<1> it) { X[it] *= -1.0f; });
      },
      P2);

  add_node(
      Graph, Queue,
      [&](handler &CGH) {
        depends_on_helper(CGH, P3);
        CGH.ext_oneapi_graph(ExecSubGraph);
      },
      P3);

  auto ExecGraph = Graph.finalize();

  auto E = Queue.submit([&](handler &CGH) { CGH.ext_oneapi_graph(ExecGraph); });

  std::vector<float> Output(N);
  Queue.memcpy(Output.data(), X, N * sizeof(float), E).wait();

  for (size_t i = 0; i < N; i++) {
    assert(Output[i] == -4.5f);
  }

  sycl::free(X, Queue);

  return 0;
}
