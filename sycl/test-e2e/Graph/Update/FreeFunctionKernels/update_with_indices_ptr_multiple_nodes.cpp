// RUN: %{build} -o %t.out
// RUN: %{run} %t.out
// Extra run to check for leaks in Level Zero using UR_L0_LEAKS_DEBUG
// RUN: %if level_zero %{env SYCL_PI_LEVEL_ZERO_USE_IMMEDIATE_COMMANDLISTS=0 %{l0_leak_check} %{run} %t.out 2>&1 | FileCheck %s --implicit-check-not=LEAK %}
// Extra run to check for immediate-command-list in Level Zero
// RUN: %if level_zero %{env SYCL_PI_LEVEL_ZERO_USE_IMMEDIATE_COMMANDLISTS=1 %{l0_leak_check} %{run} %t.out 2>&1 | FileCheck %s --implicit-check-not=LEAK %}
//
// XFAIL: cuda
// XFAIL-TRACKER: https://github.com/intel/llvm/issues/16004

// Tests updating a single dynamic parameter which is registered with multiple
// graph nodes

#include "../../graph_common.hpp"
#include "free_function_kernels.hpp"

int main() {
  queue Queue{};
  context ctxt{Queue.get_context()};

  exp_ext::command_graph Graph{Queue.get_context(), Queue.get_device()};

  int *PtrA = malloc_device<int>(Size, Queue);
  int *PtrB = malloc_device<int>(Size, Queue);

  std::vector<int> HostDataA(Size);
  std::vector<int> HostDataB(Size);

  Queue.memset(PtrA, 0, Size * sizeof(int)).wait();
  Queue.memset(PtrB, 0, Size * sizeof(int)).wait();

  exp_ext::dynamic_parameter InputParam(Graph, PtrA);

#ifndef __SYCL_DEVICE_ONLY__
  kernel_bundle Bundle = get_kernel_bundle<bundle_state::executable>(ctxt);
  kernel_id Kernel_id_A = exp_ext::get_kernel_id<ff_0>();
  kernel Kernel_A = Bundle.get_kernel(Kernel_id_A);
  auto KernelNodeA = Graph.add([&](handler &cgh) {
    cgh.set_arg(0, InputParam);
    cgh.single_task(Kernel_A);
  });

  kernel_id Kernel_id_B = exp_ext::get_kernel_id<ff_1>();
  kernel Kernel_B = Bundle.get_kernel(Kernel_id_B);
  auto KernelNodeB = Graph.add(
      [&](handler &cgh) {
        cgh.set_arg(0, InputParam);
        cgh.single_task(Kernel_B);
      },
      exp_ext::property::node::depends_on{KernelNodeA});

  auto ExecGraph = Graph.finalize(exp_ext::property::graph::updatable{});

  // PtrA should be filled with values
  Queue.ext_oneapi_graph(ExecGraph).wait();

  Queue.copy(PtrA, HostDataA.data(), Size).wait();
  Queue.copy(PtrB, HostDataB.data(), Size).wait();
  for (size_t i = 0; i < Size; i++) {
    assert(HostDataA[i] == i * 2);
    assert(HostDataB[i] == 0);
  }

  // Swap PtrB to be the input
  InputParam.update(PtrB);
  ExecGraph.update({KernelNodeA, KernelNodeB});
  Queue.ext_oneapi_graph(ExecGraph).wait();

  Queue.copy(PtrA, HostDataA.data(), Size).wait();
  Queue.copy(PtrB, HostDataB.data(), Size).wait();
  for (size_t i = 0; i < Size; i++) {
    assert(HostDataA[i] == i * 2);
    assert(HostDataB[i] == i * 2);
  }
  sycl::free(PtrA, Queue);
  sycl::free(PtrB, Queue);
#endif
  return 0;
}
