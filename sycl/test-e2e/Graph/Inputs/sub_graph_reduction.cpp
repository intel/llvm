// Modified version of the dotp example which submits which contains a sycl
// reduction as well as a sub-graph.

#include "../graph_common.hpp"

int main() {
  queue Queue;

  exp_ext::command_graph Graph{Queue.get_context(), Queue.get_device()};
  exp_ext::command_graph SubGraph{Queue.get_context(), Queue.get_device()};

  float *Dotp = malloc_device<float>(1, Queue);

  const size_t N = 10;
  float *X = malloc_device<float>(N, Queue);
  float *Y = malloc_device<float>(N, Queue);
  float *Z = malloc_device<float>(N, Queue);

  auto NodeI = add_node(Graph, Queue, [&](handler &CGH) {
    CGH.parallel_for(N, [=](id<1> it) {
      X[it] = 1.0f;
      Y[it] = 2.0f;
      Z[it] = 3.0f;
    });
  });

  auto NodeA = add_node(SubGraph, Queue, [&](handler &CGH) {
    CGH.parallel_for(range<1>{N},
                     [=](id<1> it) { X[it] = Alpha * X[it] + Beta * Y[it]; });
  });

  auto NodeB = add_node(SubGraph, Queue, [&](handler &CGH) {
    CGH.parallel_for(range<1>{N},
                     [=](id<1> it) { Z[it] = Gamma * Z[it] + Beta * Y[it]; });
  });

  auto SubGraphExec = SubGraph.finalize();

  auto NodeSub = add_node(
      Graph, Queue,
      [&](handler &CGH) {
        depends_on_helper(CGH, NodeI);
        CGH.ext_oneapi_graph(SubGraphExec);
      },
      NodeI);

  auto NodeC = add_node(
      Graph, Queue,
      [&](handler &CGH) {
        depends_on_helper(CGH, NodeSub);
        CGH.parallel_for(range<1>{N}, reduction(Dotp, 0.0f, std::plus()),
                         [=](id<1> it, auto &Sum) { Sum += X[it] * Z[it]; });
      },
      NodeSub);

  auto ExecGraph = Graph.finalize();

  // Using shortcut for executing a graph of commands
  Queue.ext_oneapi_graph(ExecGraph).wait();

  float Output;
  Queue.memcpy(&Output, Dotp, sizeof(float)).wait();
  assert(Output == dotp_reference_result(N));

  sycl::free(Dotp, Queue);
  sycl::free(X, Queue);
  sycl::free(Y, Queue);
  sycl::free(Z, Queue);

  return 0;
}
