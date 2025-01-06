// Tests using sycl_ext_oneapi_work_group_memory in a graph node with
// free functions

#include "../graph_common.hpp"
#include <sycl/ext/oneapi/experimental/work_group_memory.hpp>
#include <sycl/ext/oneapi/free_function_queries.hpp>

SYCL_EXT_ONEAPI_FUNCTION_PROPERTY(exp_ext::nd_range_kernel<1>)
void ff_local_mem(int *Ptr, exp_ext::work_group_memory<int[]> LocalMem) {
  const auto WI = sycl::ext::oneapi::this_work_item::get_nd_item<1>();
  size_t LocalID = WI.get_local_id();
  size_t GlobalID = WI.get_global_id();

  LocalMem[LocalID] = GlobalID * 2;
  Ptr[GlobalID] += LocalMem[LocalID];
}

int main() {
  queue Queue;
  exp_ext::command_graph Graph{Queue};

  std::vector<int> HostData(Size);
  std::iota(HostData.begin(), HostData.end(), 10);

  int *Ptr = malloc_device<int>(Size, Queue);
  Queue.copy(HostData.data(), Ptr, Size).wait();

  const size_t LocalSize = 128;

#ifndef __SYCL_DEVICE_ONLY__
  kernel_bundle Bundle =
      get_kernel_bundle<bundle_state::executable>(Queue.get_context());
  kernel_id Kernel_id = exp_ext::get_kernel_id<ff_local_mem>();
  kernel Kernel = Bundle.get_kernel(Kernel_id);

  auto node = add_node(Graph, Queue, [&](handler &CGH) {
    CGH.set_arg(0, Ptr);

    exp_ext::work_group_memory<int[]> WGMem{LocalSize, CGH};
    CGH.set_arg(1, WGMem);

    nd_range NDRange{{Size}, {LocalSize}};
    CGH.parallel_for(NDRange, Kernel);
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
#endif

  free(Ptr, Queue);
  return 0;
}
