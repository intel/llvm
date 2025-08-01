// XFAIL: run-mode && linux && arch-intel_gpu_bmg_g21 && spirv-backend
// XFAIL-TRACKER: https://github.com/intel/llvm/issues/19586
// RUN: %{build} -Wno-error=deprecated-declarations -o %t.out
// RUN: %{run} %t.out
// Extra run to check for leaks in Level Zero using UR_L0_LEAKS_DEBUG
// RUN: %if level_zero %{%{l0_leak_check} %{run} %t.out 2>&1 | FileCheck %s --implicit-check-not=LEAK %}

// Tests updating a graph node before finalization

#include "../../graph_common.hpp"
#include "free_function_kernels.hpp"

int main() {
  queue Queue{};
  context Ctxt{Queue.get_context()};

  exp_ext::command_graph Graph{Ctxt, Queue.get_device()};

  int *PtrA = malloc_device<int>(Size, Queue);
  int *PtrB = malloc_device<int>(Size, Queue);

  std::vector<int> HostDataA(Size);
  std::vector<int> HostDataB(Size);

  Queue.memset(PtrA, 0, Size * sizeof(int)).wait();
  Queue.memset(PtrB, 0, Size * sizeof(int)).wait();

  exp_ext::dynamic_parameter InputParam(Graph, PtrA);

  kernel_bundle Bundle = get_kernel_bundle<bundle_state::executable>(Ctxt);
  kernel_id Kernel_id = exp_ext::get_kernel_id<ff_0>();
  kernel Kernel = Bundle.get_kernel(Kernel_id);
  auto KernelNode = Graph.add([&](handler &cgh) {
    cgh.set_arg(0, InputParam);
    cgh.single_task(Kernel);
  });
  // Swap PtrB to be the input
  InputParam.update(PtrB);

  auto ExecGraph = Graph.finalize(exp_ext::property::graph::updatable{});

  // Only PtrB should be filled with values
  Queue.ext_oneapi_graph(ExecGraph).wait();

  Queue.copy(PtrA, HostDataA.data(), Size).wait();
  Queue.copy(PtrB, HostDataB.data(), Size).wait();
  for (size_t i = 0; i < Size; i++) {
    assert(HostDataA[i] == 0);
    assert(HostDataB[i] == i);
  }
  sycl::free(PtrA, Queue);
  sycl::free(PtrB, Queue);

  return 0;
}
