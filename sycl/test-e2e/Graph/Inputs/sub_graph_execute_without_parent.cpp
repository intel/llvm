// Tests creating a parent graph which contains a subgraph while also executing
// the subgraph by itself.

#include "../graph_common.hpp"

int main() {
  queue Queue{};

  if (!are_graphs_supported(Queue)) {
    return 0;
  }

  exp_ext::command_graph Graph{Queue.get_context(), Queue.get_device()};
  exp_ext::command_graph SubGraph{Queue.get_context(), Queue.get_device()};

  const size_t N = 10;
  int *X = malloc_device<int>(N, Queue);

  auto S1 = add_node(SubGraph, Queue, [&](handler &CGH) {
    CGH.parallel_for(N, [=](id<1> it) { X[it] *= 3; });
  });

  add_node(
      SubGraph, Queue,
      [&](handler &CGH) {
        depends_on_helper(CGH, S1);
        CGH.parallel_for(N, [=](id<1> it) { X[it] += 2; });
      },
      S1);

  auto ExecSubGraph = SubGraph.finalize();

  auto G1 = add_node(Graph, Queue, [&](handler &CGH) {
    CGH.parallel_for(N, [=](id<1> it) { X[it] *= 2; });
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
        CGH.parallel_for(range<1>{N}, [=](id<1> it) { X[it] *= -1; });
      },
      G2);

  auto ExecGraph = Graph.finalize();

  auto Event1 = Queue.submit(
      [&](handler &CGH) { CGH.parallel_for(N, [=](id<1> it) { X[it] = 1; }); });

  auto Event2 = Queue.submit([&](handler &CGH) {
    CGH.depends_on(Event1);
    CGH.ext_oneapi_graph(ExecSubGraph);
  });

  auto Event3 = Queue.submit([&](handler &CGH) {
    CGH.depends_on(Event2);
    CGH.ext_oneapi_graph(ExecGraph);
  });

  std::vector<int> Output(N);
  Queue.memcpy(Output.data(), X, N * sizeof(int), Event3).wait();

  const int Ref = ((1 * 3 + 2) * 2 * 3 + 2) * -1;
  for (size_t i = 0; i < N; i++) {
    assert(check_value(i, Ref, Output[i], "Output"));
  }

  sycl::free(X, Queue);

  return 0;
}
