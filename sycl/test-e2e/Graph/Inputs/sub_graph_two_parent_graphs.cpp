// Tests adding an executable graph object as a sub-graph of two different
// parent graphs.

#include "../graph_common.hpp"

int main() {
  queue Queue{};

  if (!are_graphs_supported(Queue)) {
    return 0;
  }

  exp_ext::command_graph GraphA{Queue.get_context(), Queue.get_device()};
  exp_ext::command_graph GraphB{Queue.get_context(), Queue.get_device()};
  exp_ext::command_graph SubGraph{Queue.get_context(), Queue.get_device()};

  const size_t N = 10;
  int *X = malloc_device<int>(N, Queue);

  auto S1 = add_node(SubGraph, Queue, [&](handler &CGH) {
    CGH.parallel_for(N, [=](id<1> it) { X[it] *= 2; });
  });

  add_node(
      SubGraph, Queue,
      [&](handler &CGH) {
        depends_on_helper(CGH, S1);
        CGH.parallel_for(N, [=](id<1> it) { X[it] += 1; });
      },
      S1);

  auto ExecSubGraph = SubGraph.finalize();

  auto A1 = add_node(GraphA, Queue, [&](handler &CGH) {
    CGH.parallel_for(N, [=](id<1> it) { X[it] = 1; });
  });

  auto A2 = add_node(
      GraphA, Queue,
      [&](handler &CGH) {
        depends_on_helper(CGH, A1);
        CGH.ext_oneapi_graph(ExecSubGraph);
      },
      A1);

  add_node(
      GraphA, Queue,
      [&](handler &CGH) {
        depends_on_helper(CGH, A2);
        CGH.parallel_for(range<1>{N}, [=](id<1> it) { X[it] *= -1; });
      },
      A2);

  auto ExecGraphA = GraphA.finalize();

  auto B1 = add_node(GraphB, Queue, [&](handler &CGH) {
    CGH.parallel_for(N, [=](id<1> it) { X[it] = static_cast<int>(it); });
  });

  auto B2 = add_node(
      GraphB, Queue,
      [&](handler &CGH) {
        depends_on_helper(CGH, B1);
        CGH.ext_oneapi_graph(ExecSubGraph);
      },
      B1);

  add_node(
      GraphB, Queue,
      [&](handler &CGH) {
        depends_on_helper(CGH, B2);
        CGH.parallel_for(range<1>{N}, [=](id<1> it) { X[it] *= X[it]; });
      },
      B2);

  auto ExecGraphB = GraphB.finalize();

  auto EventA1 =
      Queue.submit([&](handler &CGH) { CGH.ext_oneapi_graph(ExecGraphA); });
  std::vector<int> OutputA(N);
  auto EventA2 = Queue.memcpy(OutputA.data(), X, N * sizeof(int), EventA1);

  auto EventB1 = Queue.submit([&](handler &CGH) {
    CGH.depends_on(EventA2);
    CGH.ext_oneapi_graph(ExecGraphB);
  });
  std::vector<int> OutputB(N);
  Queue.memcpy(OutputB.data(), X, N * sizeof(int), EventB1);
  Queue.wait();

  auto refB = [](size_t i) {
    int result = static_cast<int>(i);
    result = result * 2 + 1;
    result *= result;
    return result;
  };

  const int NegThree = -3;
  for (size_t i = 0; i < N; i++) {
    assert(check_value(i, NegThree, OutputA[i], "OutputA"));
    assert(check_value(i, refB(i), OutputB[i], "OutputB"));
  }

  sycl::free(X, Queue);

  return 0;
}
