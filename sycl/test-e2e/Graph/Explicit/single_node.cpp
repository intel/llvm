// RUN: %{build} -o %t.out
// RUN: %{run} %t.out
// Extra run to check for leaks in Level Zero using UR_L0_LEAKS_DEBUG
// RUN: %if ext_oneapi_level_zero %{env UR_L0_LEAKS_DEBUG=1 %{run} %t.out 2>&1 | FileCheck %s %}
//
// CHECK-NOT: LEAK

// Tests adding a node to the graph with explicit API works as expected.

#include "../graph_common.hpp"

int main() {
  queue Queue{{sycl::ext::intel::property::queue::no_immediate_command_list{}}};

  if (!are_graphs_supported(Queue)) {
    return 0;
  }

  exp_ext::command_graph Graph{Queue.get_context(), Queue.get_device()};

  const size_t N = 10;
  int *Arr = malloc_device<int>(N, Queue);

  int ZeroPattern = 0;
  Queue.fill(Arr, ZeroPattern, N).wait();

  Graph.add([&](handler &CGH) {
    CGH.parallel_for(range<1>{N}, [=](id<1> idx) {
      size_t i = idx;
      Arr[i] = 3;
    });
  });

  std::vector<int> Output(N);
  Queue.memcpy(Output.data(), Arr, N * sizeof(int)).wait();
  int Expected = 0;
  for (size_t i = 0; i < N; i++)
    assert(check_value(i, Expected, Output[i], "Output"));

  auto ExecGraph = Graph.finalize();

  Queue.memcpy(Output.data(), Arr, N * sizeof(int)).wait();
  for (size_t i = 0; i < N; i++)
    assert(check_value(i, Expected, Output[i], "Output"));

  Queue.submit([&](handler &CGH) { CGH.ext_oneapi_graph(ExecGraph); }).wait();

  Queue.memcpy(Output.data(), Arr, N * sizeof(int)).wait();
  Expected = 3;
  for (size_t i = 0; i < N; i++)
    assert(check_value(i, Expected, Output[i], "Output"));

  sycl::free(Arr, Queue);

  return 0;
}
