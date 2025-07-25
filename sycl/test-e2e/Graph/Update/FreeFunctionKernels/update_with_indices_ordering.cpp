// XFAIL: run-mode && linux && arch-intel_gpu_bmg_g21 && spirv-backend
// XFAIL-TRACKER: https://github.com/intel/llvm/issues/19586
// RUN: %{build} -Wno-error=deprecated-declarations -o %t.out
// RUN: %{run} %t.out
// Extra run to check for leaks in Level Zero using UR_L0_LEAKS_DEBUG
// RUN: %if level_zero %{%{l0_leak_check} %{run} %t.out 2>&1 | FileCheck %s --implicit-check-not=LEAK %}

// Tests that updating a graph is ordered with respect to previous executions of
// the graph which may be in flight.

#include "../../graph_common.hpp"
#include "free_function_kernels.hpp"

int main() {
  queue Queue{};
  context Ctxt{Queue.get_context()};

  // Use a large N to try and make the kernel slow
  const size_t N = 1 << 16;

  // Reduce amount of work compared to version of test with lambdas,
  // as using explicit parameters in the free function signature results
  // in slower IR than is created from constant folding in the lambda.
  const size_t NumKernelLoops = 1;
  const size_t NumSubmitLoops = 1;

  exp_ext::command_graph Graph{Ctxt, Queue.get_device()};

  int *PtrA = malloc_device<int>(N, Queue);
  int *PtrB = malloc_device<int>(N, Queue);

  std::vector<int> HostDataA(N);
  std::vector<int> HostDataB(N);

  Queue.memset(PtrA, 0, N * sizeof(int)).wait();
  Queue.memset(PtrB, 0, N * sizeof(int)).wait();

  exp_ext::dynamic_parameter InputParam(Graph, PtrA);

  kernel_bundle Bundle = get_kernel_bundle<bundle_state::executable>(Ctxt);
  kernel_id Kernel_id = exp_ext::get_kernel_id<ff_2>();
  kernel Kernel = Bundle.get_kernel(Kernel_id);
  auto KernelNode = Graph.add([&](handler &cgh) {
    cgh.set_arg(0, InputParam);
    cgh.set_arg(1, N);
    cgh.set_arg(2, NumKernelLoops);
    cgh.single_task(Kernel);
  });

  auto ExecGraph = Graph.finalize(exp_ext::property::graph::updatable{});

  // Submit a bunch of graphs without waiting
  for (size_t i = 0; i < NumSubmitLoops; i++) {
    Queue.ext_oneapi_graph(ExecGraph);
  }

  // Swap PtrB to be the input
  InputParam.update(PtrB);

  ExecGraph.update(KernelNode);

  // Submit another set of graphs then wait on all submissions
  for (size_t i = 0; i < NumSubmitLoops; i++) {
    Queue.ext_oneapi_graph(ExecGraph);
  }
  Queue.wait_and_throw();

  Queue.copy(PtrA, HostDataA.data(), N).wait();
  Queue.copy(PtrB, HostDataB.data(), N).wait();
  for (size_t i = 0; i < N; i++) {
    assert(HostDataA[i] == i * NumKernelLoops * NumSubmitLoops);
    assert(HostDataB[i] == i * NumKernelLoops * NumSubmitLoops);
  }
  sycl::free(PtrA, Queue);
  sycl::free(PtrB, Queue);

  return 0;
}
