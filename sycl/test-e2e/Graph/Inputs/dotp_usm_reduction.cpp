// Tests constructing a graph using the explicit API to perform a dotp
// operation which uses a sycl reduction with USM memory.

#include "../graph_common.hpp"

int main() {
  queue Queue{};

  if (!are_graphs_supported(Queue)) {
    return 0;
  }

  exp_ext::command_graph Graph{Queue.get_context(), Queue.get_device()};

  int *Dotp = malloc_device<int>(1, Queue);

  const size_t N = 10;
  int *X = malloc_device<int>(N, Queue);
  int *Y = malloc_device<int>(N, Queue);
  int *Z = malloc_device<int>(N, Queue);

  auto NodeI = add_node(Graph, Queue, [&](handler &CGH) {
    CGH.parallel_for(N, [=](id<1> it) {
      X[it] = 1;
      Y[it] = 2;
      Z[it] = 3;
    });
  });

  auto NodeA = add_node(
      Graph, Queue,
      [&](handler &CGH) {
        depends_on_helper(CGH, NodeI);
        CGH.parallel_for(range<1>{N}, [=](id<1> it) {
          X[it] = Alpha * X[it] + Beta * Y[it];
        });
      },
      NodeI);

  auto NodeB = add_node(
      Graph, Queue,
      [&](handler &CGH) {
        depends_on_helper(CGH, NodeI);
        CGH.parallel_for(range<1>{N}, [=](id<1> it) {
          Z[it] = Gamma * Z[it] + Beta * Y[it];
        });
      },
      NodeI);

  auto NodeC = add_node(
      Graph, Queue,
      [&](handler &CGH) {
        depends_on_helper(CGH, {NodeA, NodeB});
        CGH.parallel_for(range<1>{N}, reduction(Dotp, 0, std::plus()),
                         [=](id<1> it, auto &Sum) { Sum += X[it] * Z[it]; });
      },
      NodeA, NodeB);

  auto ExecGraph = Graph.finalize();

  // Using shortcut for executing a graph of commands
  Queue.ext_oneapi_graph(ExecGraph).wait();

  int Output;
  Queue.memcpy(&Output, Dotp, sizeof(int)).wait();

  assert(Output == dotp_reference_result(N));

  sycl::free(Dotp, Queue);
  sycl::free(X, Queue);
  sycl::free(Y, Queue);
  sycl::free(Z, Queue);

  return 0;
}
