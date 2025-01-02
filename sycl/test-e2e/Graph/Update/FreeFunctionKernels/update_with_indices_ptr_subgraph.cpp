// RUN: %{build} -o %t.out
// RUN: %{run} %t.out
// Extra run to check for leaks in Level Zero using UR_L0_LEAKS_DEBUG
// RUN: %if level_zero %{env SYCL_PI_LEVEL_ZERO_USE_IMMEDIATE_COMMANDLISTS=0 %{l0_leak_check} %{run} %t.out 2>&1 | FileCheck %s --implicit-check-not=LEAK %}
// Extra run to check for immediate-command-list in Level Zero
// RUN: %if level_zero %{env SYCL_PI_LEVEL_ZERO_USE_IMMEDIATE_COMMANDLISTS=1 %{l0_leak_check} %{run} %t.out 2>&1 | FileCheck %s --implicit-check-not=LEAK %}
//
// XFAIL: cuda
// XFAIL-TRACKER: https://github.com/intel/llvm/issues/16004

// Tests updating a graph node in an executable graph that was used as a
// subgraph node in another executable graph is not reflected in the graph
// containing the subgraph node.

#include "../../graph_common.hpp"
#include "free_function_kernels.hpp"

int main() {
  queue Queue{};
  context Ctxt{Queue.get_context()};

  exp_ext::command_graph Graph{Ctxt, Queue.get_device()};
  exp_ext::command_graph SubGraph{Ctxt, Queue.get_device()};

  int *PtrA = malloc_device<int>(Size, Queue);
  int *PtrB = malloc_device<int>(Size, Queue);

  std::vector<int> HostDataA(Size);
  std::vector<int> HostDataB(Size);

  Queue.memset(PtrA, 0, Size * sizeof(int)).wait();
  Queue.memset(PtrB, 0, Size * sizeof(int)).wait();

  exp_ext::dynamic_parameter InputParam(SubGraph, PtrA);

#ifndef __SYCL_DEVICE_ONLY__
  kernel_bundle Bundle = get_kernel_bundle<bundle_state::executable>(Ctxt);
  kernel_id SubKernel_id = exp_ext::get_kernel_id<ff_1>();
  kernel SubKernel = Bundle.get_kernel(SubKernel_id);
  auto SubKernelNode = SubGraph.add([&](handler &cgh) {
    cgh.set_arg(0, InputParam);
    cgh.single_task(SubKernel);
  });

  auto SubExecGraph = SubGraph.finalize(exp_ext::property::graph::updatable{});

  kernel_id Kernel_id = exp_ext::get_kernel_id<ff_0>();
  kernel Kernel = Bundle.get_kernel(Kernel_id);
  auto KernelNode = Graph.add([&](handler &cgh) {
    cgh.set_arg(0, PtrA);
    cgh.single_task(Kernel);
  });

  Graph.add([&](handler &cgh) { cgh.ext_oneapi_graph(SubExecGraph); },
            exp_ext::property::node::depends_on{KernelNode});

  // Finalize the parent graph with the original values
  auto ExecGraph = Graph.finalize();

  // Swap PtrB to be the input
  InputParam.update(PtrB);
  // Update the executable graph that was used as a subgraph with the new value,
  // this should not affect ExecGraph
  SubExecGraph.update(SubKernelNode);
  // Only PtrA should be filled with values
  Queue.ext_oneapi_graph(ExecGraph).wait();

  Queue.copy(PtrA, HostDataA.data(), Size).wait();
  Queue.copy(PtrB, HostDataB.data(), Size).wait();
  for (size_t i = 0; i < Size; i++) {
    assert(HostDataA[i] == i * 2);
    assert(HostDataB[i] == 0);
  }
#endif
  sycl::free(PtrA, Queue);
  sycl::free(PtrB, Queue);

  return 0;
}
