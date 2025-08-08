// Tests using sycl_ext_oneapi_work_group_memory in a graph node

#include "../graph_common.hpp"
#include <sycl/ext/oneapi/experimental/work_group_memory.hpp>

int main() {
  queue Queue;
  exp_ext::command_graph Graph{Queue};

  std::vector<int> HostData(Size);
  std::iota(HostData.begin(), HostData.end(), 10);

  int *Ptr = malloc_device<int>(Size, Queue);
  Queue.copy(HostData.data(), Ptr, Size).wait();

  const size_t LocalSize = 128;
  auto node = add_node(Graph, Queue, [&](handler &CGH) {
    exp_ext::work_group_memory<int[]> WGMem{LocalSize, CGH};

    CGH.parallel_for(nd_range({Size}, {LocalSize}), [=](nd_item<1> Item) {
      WGMem[Item.get_local_linear_id()] = Item.get_global_linear_id() * 2;
      Ptr[Item.get_global_linear_id()] += WGMem[Item.get_local_linear_id()];
    });
  });

  auto GraphExec = Graph.finalize();

  for (unsigned N = 0; N < Iterations; N++) {
    Queue.submit([&](handler &CGH) { CGH.ext_oneapi_graph(GraphExec); });
  }
  Queue.wait_and_throw();

  Queue.copy(Ptr, HostData.data(), Size);
  Queue.wait_and_throw();

  for (size_t i = 0; i < Size; i++) {
    int Ref = 10 + i + (Iterations * (i * 2));
    assert(check_value(i, Ref, HostData[i], "Ptr"));
  }

  free(Ptr, Queue);
  return 0;
}
