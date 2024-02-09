// Test executing a graph multiple times.

#include "../graph_common.hpp"

int main() {
  queue Queue{};

  if (!are_graphs_supported(Queue)) {
    return 0;
  }

  exp_ext::command_graph Graph{Queue.get_context(), Queue.get_device()};

  const size_t N = 10;
  int *Arr = malloc_device<int>(N, Queue);
  int ZeroPattern = 0;
  Queue.fill(Arr, ZeroPattern, N).wait();

  add_node(Graph, Queue, [&](handler &CGH) {
    CGH.parallel_for(range<1>{N}, [=](id<1> idx) {
      size_t i = idx;
      Arr[i] += 1;
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
  Expected = 1;
  for (size_t i = 0; i < N; i++)
    assert(check_value(i, Expected, Output[i], "Output"));

  Queue.submit([&](handler &CGH) { CGH.ext_oneapi_graph(ExecGraph); }).wait();

  Queue.memcpy(Output.data(), Arr, N * sizeof(int)).wait();
  Expected = 2;
  for (size_t i = 0; i < N; i++)
    assert(check_value(i, Expected, Output[i], "Output"));

  sycl::free(Arr, Queue);

  return 0;
}
