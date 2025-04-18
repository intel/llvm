// Tests using sycl_ext_oneapi_work_group_static in a graph node

#include "../graph_common.hpp"
#include <sycl/ext/oneapi/work_group_static.hpp>

constexpr size_t WgSize = 32;

// Local mem used in kernel
sycl::ext::oneapi::experimental::work_group_static<int[WgSize]> LocalIDBuff;

int main() {
  queue Queue;
  exp_ext::command_graph Graph{Queue};

  std::vector<int> HostData(Size, 0);

  int *Ptr = malloc_device<int>(Size, Queue);
  Queue.copy(HostData.data(), Ptr, Size).wait();

  auto node = add_node(Graph, Queue, [&](handler &CGH) {
    CGH.parallel_for(nd_range({Size}, {WgSize}), [=](nd_item<1> Item) {
      LocalIDBuff[Item.get_local_linear_id()] = Item.get_local_linear_id();

      Item.barrier();

      // Check that the memory is accessible from other work-items
      size_t LocalIdx = Item.get_local_linear_id() ^ 1;
      size_t GlobalIdx = Item.get_global_linear_id() ^ 1;
      Ptr[GlobalIdx] = LocalIDBuff[LocalIdx];
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
    int Ref = i % WgSize;
    assert(check_value(i, Ref, HostData[i], "Ptr"));
  }

  free(Ptr, Queue);
  return 0;
}
