// Tests creating a parent graph which contains a subgraph while also executing
// the subgraph by itself.

#include "../graph_common.hpp"

int main() {
  queue Queue;

  exp_ext::command_graph Graph{Queue.get_context(), Queue.get_device()};
  exp_ext::command_graph SubGraph{Queue.get_context(), Queue.get_device()};

  const size_t N = 10;
  float *X = malloc_device<float>(N, Queue);

  auto S1 = add_node(SubGraph, Queue, [&](handler &CGH) {
    CGH.parallel_for(N, [=](id<1> it) { X[it] *= 3.14f; });
  });

  add_node(
      SubGraph, Queue,
      [&](handler &CGH) {
        depends_on_helper(CGH, S1);
        CGH.parallel_for(N, [=](id<1> it) { X[it] += 0.5f; });
      },
      S1);

  auto ExecSubGraph = SubGraph.finalize();

  auto G1 = add_node(Graph, Queue, [&](handler &CGH) {
    CGH.parallel_for(N, [=](id<1> it) { X[it] *= 2.0f; });
  });

  auto G2 = add_node(
      Graph, Queue,
      [&](handler &CGH) {
        depends_on_helper(CGH, G1);
        CGH.ext_oneapi_graph(ExecSubGraph);
      },
      G1);

  add_node(
      Graph, Queue,
      [&](handler &CGH) {
        depends_on_helper(CGH, G2);
        CGH.parallel_for(range<1>{N}, [=](id<1> it) { X[it] *= -1.0f; });
      },
      G2);

  auto ExecGraph = Graph.finalize();

  auto Event1 = Queue.submit([&](handler &CGH) {
    CGH.parallel_for(N, [=](id<1> it) { X[it] = 1.f; });
  });

  auto Event2 = Queue.submit([&](handler &CGH) {
    CGH.depends_on(Event1);
    CGH.ext_oneapi_graph(ExecSubGraph);
  });

  auto Event3 = Queue.submit([&](handler &CGH) {
    CGH.depends_on(Event2);
    CGH.ext_oneapi_graph(ExecGraph);
  });

  std::vector<float> Output(N);
  Queue.memcpy(Output.data(), X, N * sizeof(float), Event3).wait();

  const float ref = ((1.f * 3.14f + 0.5f) * 2.0f * 3.14f + 0.5f) * -1.f;
  for (size_t i = 0; i < N; i++) {
    assert(Output[i] == ref);
  }

  sycl::free(X, Queue);

  return 0;
}
