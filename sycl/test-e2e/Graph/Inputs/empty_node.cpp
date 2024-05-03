// Tests the interface for adding empty nodes and creating dependencies on those
// empty nodes.

#include "../graph_common.hpp"

int main() {
  queue Queue{};

  auto MyProperties = property_list{exp_ext::property::graph::no_cycle_check()};
  exp_ext::command_graph Graph{Queue.get_context(), Queue.get_device(),
                               MyProperties};

  const size_t N = 10;
  int *Arr = malloc_device<int>(N, Queue);

  auto Start = add_empty_node(Graph, Queue);

  auto Init = add_node(
      Graph, Queue,
      [&](handler &CGH) {
        depends_on_helper(CGH, Start);
        CGH.parallel_for(range<1>{N}, [=](id<1> idx) {
          size_t i = idx;
          Arr[i] = 0;
        });
      },
      Start);

  auto Empty = add_empty_node(Graph, Queue, Init);
  auto Empty2 = add_empty_node(Graph, Queue, Empty);
  auto Empty3 = add_node(
      Graph, Queue, [&](handler &CGH) { depends_on_helper(CGH, Empty2); },
      Empty2);

  add_node(
      Graph, Queue,
      [&](handler &CGH) {
        depends_on_helper(CGH, Empty2);
        CGH.parallel_for(range<1>{N}, [=](id<1> idx) {
          size_t i = idx;
          Arr[i] = 1;
        });
      },
      Empty2);

  auto ExecGraph = Graph.finalize();

  Queue.submit([&](handler &CGH) { CGH.ext_oneapi_graph(ExecGraph); }).wait();

  std::vector<int> HostData(N);
  Queue.memcpy(HostData.data(), Arr, N * sizeof(int)).wait();

  const int Expected = 1;
  for (size_t i = 0; i < N; i++)
    assert(check_value(i, Expected, HostData[i], "HostData"));

  sycl::free(Arr, Queue);

  return 0;
}
