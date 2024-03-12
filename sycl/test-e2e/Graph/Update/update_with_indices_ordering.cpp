// RUN: %{build} -o %t.out
// RUN: %{run} %t.out
// Extra run to check for leaks in Level Zero using UR_L0_LEAKS_DEBUG
// RUN: %if level_zero %{env SYCL_PI_LEVEL_ZERO_USE_IMMEDIATE_COMMANDLISTS=0 %{l0_leak_check} %{run} %t.out 2>&1 | FileCheck %s --implicit-check-not=LEAK %}
// Extra run to check for immediate-command-list in Level Zero
// RUN: %if level_zero && linux %{env SYCL_PI_LEVEL_ZERO_USE_IMMEDIATE_COMMANDLISTS=1 %{l0_leak_check} %{run} %t.out 2>&1 | FileCheck %s --implicit-check-not=LEAK %}
//
// UNSUPPORTED: opencl, level_zero

// Tests that updating a graph is ordered with respect to previous executions of
// the graph which may be in flight.

#include "../graph_common.hpp"

int main() {
  queue Queue{};

  // Use a large N to try and make the kernel slow
  const size_t N = 1 << 16;
  // Loop inside kernel to make even slower (too large N runs out of memory)
  const size_t NumKernelLoops = 4;
  const size_t NumSubmitLoops = 8;

  exp_ext::command_graph Graph{Queue.get_context(), Queue.get_device()};

  int *PtrA = malloc_device<int>(N, Queue);
  int *PtrB = malloc_device<int>(N, Queue);

  std::vector<int> HostDataA(N);
  std::vector<int> HostDataB(N);

  Queue.memset(PtrA, 0, N * sizeof(int)).wait();
  Queue.memset(PtrB, 0, N * sizeof(int)).wait();

  exp_ext::dynamic_parameter InputParam(Graph, PtrA);

  auto KernelNode = Graph.add([&](handler &cgh) {
    cgh.set_arg(0, InputParam);
    // TODO: Use the free function kernel extension instead of regular kernels
    // when available.
    cgh.single_task([=]() {
      for (size_t j = 0; j < NumKernelLoops; j++) {
        for (size_t i = 0; i < N; i++) {
          PtrA[i] += i;
        }
      }
    });
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
  return 0;
}
