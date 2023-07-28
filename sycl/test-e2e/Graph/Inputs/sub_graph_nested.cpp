// This tests nesting sub-graphs two deep inside a parent graph.

#include "../graph_common.hpp"

namespace {
// Calculates reference result at index i
float reference(size_t i) {
  float x = static_cast<float>(i);
  float y = static_cast<float>(i);
  float z = static_cast<float>(i);

  x = x * 2.0f + 0.5f;  // XSubSubGraph
  y = y * 3.0f + 0.14f; // YSubSubGraph

  // SubGraph
  x = -x;
  y = -y;

  // Graph
  z = z * x - y;

  return z;
}
} // namespace

int main() {
  queue Queue;

  exp_ext::command_graph Graph{Queue.get_context(), Queue.get_device()};
  exp_ext::command_graph SubGraph{Queue.get_context(), Queue.get_device()};
  exp_ext::command_graph XSubSubGraph{Queue.get_context(), Queue.get_device()};
  exp_ext::command_graph YSubSubGraph{Queue.get_context(), Queue.get_device()};

  const size_t N = 10;
  float *X = malloc_device<float>(N, Queue);
  float *Y = malloc_device<float>(N, Queue);
  float *Z = malloc_device<float>(N, Queue);

  // XSubSubGraph is a multiply-add operation on USM allocation X
  auto XSS1 = add_node(XSubSubGraph, Queue, [&](handler &CGH) {
    CGH.parallel_for(N, [=](id<1> it) {
      const size_t i = it[0];
      X[i] *= 2.0f;
    });
  });

  add_node(
      XSubSubGraph, Queue,
      [&](handler &CGH) {
        depends_on_helper(CGH, XSS1);
        CGH.parallel_for(N, [=](id<1> it) {
          const size_t i = it[0];
          X[i] += 0.5f;
        });
      },
      XSS1);

  auto XExecSubSubGraph = XSubSubGraph.finalize();

  // YSubSubGraph is a multiply-add operation on USM allocation Y
  auto YSS1 = add_node(YSubSubGraph, Queue, [&](handler &CGH) {
    CGH.parallel_for(N, [=](id<1> it) {
      const size_t i = it[0];
      Y[i] *= 3.0f;
    });
  });

  add_node(
      YSubSubGraph, Queue,
      [&](handler &CGH) {
        depends_on_helper(CGH, YSS1);
        CGH.parallel_for(N, [=](id<1> it) {
          const size_t i = it[0];
          Y[i] += 0.14f;
        });
      },
      YSS1);

  auto YExecSubSubGraph = YSubSubGraph.finalize();

  // SubGraph initializes X & Y inputs, adds both subgraphs, then negates
  // the results
  auto S1 = add_node(SubGraph, Queue, [&](handler &CGH) {
    CGH.parallel_for(N, [=](id<1> it) {
      const size_t i = it[0];
      X[i] = static_cast<float>(i);
      Y[i] = static_cast<float>(i);
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
          const size_t i = it[0];
          X[i] = -X[i];
          Y[i] = -Y[i];
        });
      },
      S2, S3);

  auto ExecSubGraph = SubGraph.finalize();

  // Parent Graph initializes Z allocation, adds the sub-graph,then
  // does a multiply add with X & Y allocation results.
  auto G1 = add_node(Graph, Queue, [&](handler &CGH) {
    CGH.parallel_for(range<1>{N}, [=](id<1> it) {
      const size_t i = it[0];
      Z[i] = static_cast<float>(i);
    });
  });

  auto G2 = add_node(Graph, Queue,
                     [&](handler &CGH) { CGH.ext_oneapi_graph(ExecSubGraph); });

  add_node(
      Graph, Queue,
      [&](handler &CGH) {
        depends_on_helper(CGH, {G1, G2});
        CGH.parallel_for(range<1>{N}, [=](id<1> it) {
          const size_t i = it[0];
          Z[i] = Z[i] * X[i] - Y[i];
        });
      },
      G1, G2);

  auto ExecGraph = Graph.finalize();

  auto E = Queue.submit([&](handler &CGH) { CGH.ext_oneapi_graph(ExecGraph); });

  std::vector<float> Output(N);
  Queue.memcpy(Output.data(), Z, N * sizeof(float), E).wait();

  for (size_t i = 0; i < N; i++) {
    float ref = reference(i);
    assert(Output[i] == ref);
  }

  sycl::free(X, Queue);
  sycl::free(Y, Queue);
  sycl::free(Z, Queue);

  return 0;
}
