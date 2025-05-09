// RUN: %{build} -Wno-error=deprecated-declarations -o %t.out
// RUN: %{run} %t.out
// Extra run to check for leaks in Level Zero using UR_L0_LEAKS_DEBUG
// RUN: %if level_zero %{env SYCL_PI_LEVEL_ZERO_USE_IMMEDIATE_COMMANDLISTS=0 %{l0_leak_check} %{run} %t.out 2>&1 | FileCheck %s --implicit-check-not=LEAK %}
// Extra run to check for immediate-command-list in Level Zero
// RUN: %if level_zero %{env SYCL_PI_LEVEL_ZERO_USE_IMMEDIATE_COMMANDLISTS=1 %{l0_leak_check} %{run} %t.out 2>&1 | FileCheck %s --implicit-check-not=LEAK %}

// Tests updating dynamic_work_group_memory with a new size.

#include "../../graph_common.hpp"
#include "free_function_kernels.hpp"

int main() {
  constexpr int LocalSize{32};
  nd_range<1> NDRange{Size, LocalSize};

  queue Queue{};
  context Ctxt{Queue.get_context()};

  int *PtrA = malloc_device<int>(Size, Queue);
  std::vector<int> HostDataA(Size);

  exp_ext::command_graph Graph{Ctxt, Queue.get_device()};
  exp_ext::dynamic_work_group_memory<int[]> DynLocalMem{Graph, LocalSize};

  Queue.memset(PtrA, 0, Size * sizeof(int)).wait();

  kernel_bundle Bundle = get_kernel_bundle<bundle_state::executable>(Ctxt);
  kernel_id KernelID = exp_ext::get_kernel_id<ff_7>();
  kernel Kernel = Bundle.get_kernel(KernelID);
  auto KernelNode = Graph.add([&](handler &cgh) {
    cgh.set_arg(0, DynLocalMem);
    cgh.set_arg(1, PtrA);
    cgh.parallel_for(NDRange, Kernel);
  });

  auto ExecGraph = Graph.finalize(exp_ext::property::graph::updatable{});
  Queue.ext_oneapi_graph(ExecGraph).wait();

  Queue.copy(PtrA, HostDataA.data(), Size).wait();
  for (size_t i = 0; i < Size; i++) {
    assert(HostDataA[i] == LocalSize * LocalSize);
  }

  constexpr int NewLocalSize{64};
  DynLocalMem.update(NewLocalSize);
  KernelNode.update_nd_range(nd_range<1>{range{Size}, range{NewLocalSize}});
  ExecGraph.update(KernelNode);

  Queue.memset(PtrA, 0, Size * sizeof(int)).wait();
  Queue.ext_oneapi_graph(ExecGraph).wait();

  Queue.copy(PtrA, HostDataA.data(), Size).wait();
  for (size_t i = 0; i < Size; i++) {
    assert(HostDataA[i] == NewLocalSize * NewLocalSize);
  }

  free(PtrA, Queue);
  return 0;
}
