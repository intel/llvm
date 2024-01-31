// RUN: %{build} -o %t.out
// RUN: %{run} %t.out
// Extra run to check for leaks in Level Zero using UR_L0_LEAKS_DEBUG
// RUN: %if (level_zero && linux) %{env UR_L0_LEAKS_DEBUG=1 %{run} %t.out 2>&1 | FileCheck %s %}
// RUN: %if (level_zero && windows) %{env UR_L0_LEAKS_DEBUG=1 env SYCL_ENABLE_DEFAULT_CONTEXTS=0 %{run} %t.out 2>&1 | FileCheck %s %}
//
// CHECK-NOT: LEAK

// Tests that graph.make_edge() correctly defines the dependency between two
// nodes.

#include "../graph_common.hpp"

int main() {
  queue Queue{{sycl::ext::intel::property::queue::no_immediate_command_list{}}};

  if (!are_graphs_supported(Queue)) {
    return 0;
  }

  exp_ext::command_graph Graph{Queue.get_context(), Queue.get_device()};

  const size_t N = 10;
  int *X = malloc_device<int>(N, Queue);

  auto Init = Graph.add([&](handler &CGH) {
    CGH.parallel_for(range<1>{N}, [=](id<1> idx) { X[idx] = 2; });
  });

  auto Add = Graph.add([&](handler &CGH) {
    CGH.parallel_for(range<1>{N}, [=](id<1> idx) { X[idx] += 2; });
  });

  auto Mult = Graph.add([&](handler &CGH) {
    CGH.parallel_for(range<1>{N}, [=](id<1> idx) { X[idx] *= 3; });
  });

  Graph.make_edge(Init, Mult);
  Graph.make_edge(Mult, Add);

  auto ExecGraph = Graph.finalize();

  Queue.submit([&](handler &CGH) { CGH.ext_oneapi_graph(ExecGraph); }).wait();

  std::vector<int> Output(N);
  Queue.memcpy(Output.data(), X, N * sizeof(int)).wait();

  const int Expected = 8;
  for (size_t i = 0; i < N; i++)
    assert(check_value(i, Expected, Output[i], "Output"));

  sycl::free(X, Queue);

  return 0;
}
