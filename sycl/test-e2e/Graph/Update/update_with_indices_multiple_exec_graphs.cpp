// RUN: %{build} -o %t.out
// RUN: %{run} %t.out
// Extra run to check for leaks in Level Zero using UR_L0_LEAKS_DEBUG
// RUN: %if level_zero %{env SYCL_PI_LEVEL_ZERO_USE_IMMEDIATE_COMMANDLISTS=0 %{l0_leak_check} %{run} %t.out 2>&1 | FileCheck %s --implicit-check-not=LEAK %}
// Extra run to check for immediate-command-list in Level Zero
// RUN: %if level_zero && linux %{env SYCL_PI_LEVEL_ZERO_USE_IMMEDIATE_COMMANDLISTS=1 %{l0_leak_check} %{run} %t.out 2>&1 | FileCheck %s --implicit-check-not=LEAK %}
//
// UNSUPPORTED: opencl, level_zero

// Tests creating multiple executable graphs from the same modifiable graph and
// only updating one of them.

#include "../graph_common.hpp"

int main() {
  queue Queue{};

  const size_t N = 1024;

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
      for (size_t i = 0; i < N; i++) {
        PtrA[i] += i;
      }
    });
  });

  auto ExecGraph = Graph.finalize(exp_ext::property::graph::updatable{});
  auto ExecGraph2 = Graph.finalize(exp_ext::property::graph::updatable{});

  // PtrA values should be modified twice
  Queue.ext_oneapi_graph(ExecGraph).wait();
  Queue.ext_oneapi_graph(ExecGraph2).wait();

  Queue.copy(PtrA, HostDataA.data(), N).wait();
  Queue.copy(PtrB, HostDataB.data(), N).wait();
  for (size_t i = 0; i < N; i++) {
    assert(HostDataA[i] == i * 2);
    assert(HostDataB[i] == 0);
  }

  // Swap PtrB to be the input
  InputParam.update(PtrB);
  // Only update ExecGraph, which should now modify PtrB while ExecGraph2
  // modifies PtrA still
  ExecGraph.update(KernelNode);
  Queue.ext_oneapi_graph(ExecGraph).wait();
  Queue.ext_oneapi_graph(ExecGraph2).wait();

  Queue.copy(PtrA, HostDataA.data(), N).wait();
  Queue.copy(PtrB, HostDataB.data(), N).wait();
  for (size_t i = 0; i < N; i++) {
    // A should have been modified 3 times by now, B only once
    assert(HostDataA[i] == i * 3);
    assert(HostDataB[i] == i);
  }
  return 0;
}
