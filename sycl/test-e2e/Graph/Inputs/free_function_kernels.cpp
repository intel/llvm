// Tests compatibility with free function kernels extension

#include "../graph_common.hpp"

SYCL_EXT_ONEAPI_FUNCTION_PROPERTY((exp_ext::single_task_kernel))
void ff_0(int *ptr) {
  for (size_t i{0}; i < Size; ++i) {
    ptr[i] = i;
  }
}

int main() {
  queue Queue{};
  context ctxt{Queue.get_context()};

  exp_ext::command_graph Graph{ctxt, Queue.get_device()};

  int *PtrA = malloc_device<int>(Size, Queue);

  std::vector<int> HostDataA(Size);

  Queue.memset(PtrA, 0, Size * sizeof(int)).wait();

#ifndef __SYCL_DEVICE_ONLY__
  kernel_bundle Bundle = get_kernel_bundle<bundle_state::executable>(ctxt);
  kernel_id Kernel_id = exp_ext::get_kernel_id<ff_0>();
  kernel Kernel = Bundle.get_kernel(Kernel_id);
  auto KernelNode = Graph.add([&](handler &cgh) {
    cgh.set_arg(0, PtrA);
    cgh.single_task(Kernel);
  });

  auto ExecGraph = Graph.finalize();

  Queue.ext_oneapi_graph(ExecGraph).wait();

  Queue.copy(PtrA, HostDataA.data(), Size).wait();
  for (size_t i = 0; i < Size; i++) {
    assert(HostDataA[i] == i);
  }
  sycl::free(PtrA, Queue);
#endif
  return 0;
}
