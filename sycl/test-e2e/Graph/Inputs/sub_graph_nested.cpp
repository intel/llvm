// This tests nesting sub-graphs two deep inside a parent graph.

#include "../graph_common.hpp"

namespace {
// Calculates reference result at index i
int reference(size_t i) {
  int x = static_cast<int>(i);
  int y = static_cast<int>(i);
  int z = static_cast<int>(i);

  x = x * 2 + 0.5f;  // XSubSubGraph
  y = y * 3 + 0.14f; // YSubSubGraph

  // SubGraph
  x = -x;
  y = -y;

  // Graph
  z = z * x - y;

  return z;
}
} // namespace

int main() {
  queue Queue{};

  exp_ext::command_graph Graph{Queue.get_context(), Queue.get_device()};
  exp_ext::command_graph SubGraph{Queue.get_context(), Queue.get_device()};
  exp_ext::command_graph XSubSubGraph{Queue.get_context(), Queue.get_device()};
  exp_ext::command_graph YSubSubGraph{Queue.get_context(), Queue.get_device()};

  const size_t N = 10;
  int *X = malloc_device<int>(N, Queue);
  int *Y = malloc_device<int>(N, Queue);
  int *Z = malloc_device<int>(N, Queue);

  // XSubSubGraph is a multiply-add operation on USM allocation X
  auto XSS1 = add_node(XSubSubGraph, Queue, [&](handler &CGH) {
    CGH.parallel_for(N, [=](id<1> it) { X[it] *= 2; });
  });

  add_node(
      XSubSubGraph, Queue,
      [&](handler &CGH) {
        depends_on_helper(CGH, XSS1);
        CGH.parallel_for(N, [=](id<1> it) { X[it] += 0.5f; });
      },
      XSS1);

  auto XExecSubSubGraph = XSubSubGraph.finalize();

  // YSubSubGraph is a multiply-add operation on USM allocation Y
  auto YSS1 = add_node(YSubSubGraph, Queue, [&](handler &CGH) {
    CGH.parallel_for(N, [=](id<1> it) { Y[it] *= 3; });
  });

  add_node(
      YSubSubGraph, Queue,
      [&](handler &CGH) {
        depends_on_helper(CGH, YSS1);
        CGH.parallel_for(N, [=](id<1> it) { Y[it] += 0.14f; });
      },
      YSS1);

  auto YExecSubSubGraph = YSubSubGraph.finalize();

  // SubGraph initializes X & Y inputs, adds both subgraphs, then negates
  // the results
  auto S1 = add_node(SubGraph, Queue, [&](handler &CGH) {
    CGH.parallel_for(N, [=](id<1> it) {
      X[it] = static_cast<int>(it);
      Y[it] = static_cast<int>(it);
    });
  });

  auto S2 = add_node(
      SubGraph, Queue,
      [&](handler &CGH) {
        depends_on_helper(CGH, S1);
        CGH.ext_oneapi_graph(XExecSubSubGraph);
      },
      S1);

  auto S3 = add_node(
      SubGraph, Queue,
      [&](handler &CGH) {
        depends_on_helper(CGH, S1);
        CGH.ext_oneapi_graph(YExecSubSubGraph);
      },
      S1);

  add_node(
      SubGraph, Queue,
      [&](handler &CGH) {
        depends_on_helper(CGH, {S2, S3});
        CGH.parallel_for(N, [=](id<1> it) {
          X[it] = -X[it];
          Y[it] = -Y[it];
        });
      },
      S2, S3);

  auto ExecSubGraph = SubGraph.finalize();

  // Parent Graph initializes Z allocation, adds the sub-graph,then
  // does a multiply add with X & Y allocation results.
  auto G1 = add_node(Graph, Queue, [&](handler &CGH) {
    CGH.parallel_for(range<1>{N},
                     [=](id<1> it) { Z[it] = static_cast<int>(it); });
  });

  auto G2 = add_node(Graph, Queue,
                     [&](handler &CGH) { CGH.ext_oneapi_graph(ExecSubGraph); });

  add_node(
      Graph, Queue,
      [&](handler &CGH) {
        depends_on_helper(CGH, {G1, G2});
        CGH.parallel_for(range<1>{N},
                         [=](id<1> it) { Z[it] = Z[it] * X[it] - Y[it]; });
      },
      G1, G2);

  auto ExecGraph = Graph.finalize();

  auto E = Queue.submit([&](handler &CGH) { CGH.ext_oneapi_graph(ExecGraph); });

  std::vector<int> Output(N);
  Queue.memcpy(Output.data(), Z, N * sizeof(int), E).wait();

  for (size_t i = 0; i < N; i++) {
    int Ref = reference(i);
    assert(check_value(i, Ref, Output[i], "Output"));
  }

  sycl::free(X, Queue);
  sycl::free(Y, Queue);
  sycl::free(Z, Queue);

  return 0;
}
