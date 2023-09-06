// Test executing a graph multiple times.

#include "../graph_common.hpp"

int main() {

  queue Queue{{sycl::ext::intel::property::queue::no_immediate_command_list{}}};

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
  for (int i = 0; i < N; i++)
    assert(Output[i] == 0);

  auto ExecGraph = Graph.finalize();

  Queue.memcpy(Output.data(), Arr, N * sizeof(int)).wait();
  for (int i = 0; i < N; i++)
    assert(Output[i] == 0);

  Queue.submit([&](handler &CGH) { CGH.ext_oneapi_graph(ExecGraph); }).wait();

  Queue.memcpy(Output.data(), Arr, N * sizeof(int)).wait();
  for (int i = 0; i < N; i++)
    assert(Output[i] == 1);

  Queue.submit([&](handler &CGH) { CGH.ext_oneapi_graph(ExecGraph); }).wait();

  Queue.memcpy(Output.data(), Arr, N * sizeof(int)).wait();
  for (int i = 0; i < N; i++)
    assert(Output[i] == 2);

  sycl::free(Arr, Queue);

  return 0;
}
