// RUN: %{build} -o %t.out
// RUN: %{run} %t.out
// Extra run to check for leaks in Level Zero using UR_L0_LEAKS_DEBUG
// RUN: %if level_zero %{env SYCL_PI_LEVEL_ZERO_USE_IMMEDIATE_COMMANDLISTS=0 %{l0_leak_check} %{run} %t.out 2>&1 | FileCheck %s --implicit-check-not=LEAK %}
// Extra run to check for immediate-command-list in Level Zero
// RUN: %if level_zero && linux %{env SYCL_PI_LEVEL_ZERO_USE_IMMEDIATE_COMMANDLISTS=1 %{l0_leak_check} %{run} %t.out 2>&1 | FileCheck %s --implicit-check-not=LEAK %}
//
// UNSUPPORTED: opencl, level_zero

// Tests updating multiple parameters to a singlegraph node using index-based
// explicit update

#include "../graph_common.hpp"

int main() {
  queue Queue{};

  const size_t N = 1024;

  exp_ext::command_graph Graph{Queue.get_context(), Queue.get_device()};

  int *PtrA = malloc_device<int>(N, Queue);
  int *PtrB = malloc_device<int>(N, Queue);
  int *PtrC = malloc_device<int>(N, Queue);

  std::vector<int> HostDataA(N);
  std::vector<int> HostDataB(N);
  std::vector<int> HostDataC(N);
  std::vector<int> OutData(N);

  std::iota(HostDataA.begin(), HostDataA.end(), 10);
  std::iota(HostDataB.begin(), HostDataB.end(), 100);

  Queue.memcpy(PtrA, HostDataA.data(), N * sizeof(int)).wait();
  Queue.memcpy(PtrB, HostDataB.data(), N * sizeof(int)).wait();
  Queue.memset(PtrC, 0, N * sizeof(int)).wait();

  exp_ext::dynamic_parameter ParamA(Graph, PtrA);
  exp_ext::dynamic_parameter ParamB(Graph, PtrB);
  exp_ext::dynamic_parameter ParamOut(Graph, PtrC);

  auto KernelNode = Graph.add([&](handler &cgh) {
    cgh.set_arg(0, ParamOut);
    cgh.set_arg(1, ParamA);
    cgh.set_arg(2, ParamB);
    // TODO: Use the free function kernel extension instead of regular kernels
    // when available.
    cgh.parallel_for(range<1>{Size}, [=](item<1> Item) {
      size_t ID = Item.get_id();
      PtrC[ID] += PtrA[ID] * PtrB[ID];
    });
  });

  auto ExecGraph = Graph.finalize(exp_ext::property::graph::updatable{});

  // PtrA should be filled with values
  Queue.ext_oneapi_graph(ExecGraph).wait();

  // Copy to output data to preserve original data for verifying += op
  Queue.copy(PtrC, OutData.data(), N).wait();
  for (size_t i = 0; i < N; i++) {
    assert(OutData[i] == HostDataC[i] + (HostDataA[i] * HostDataB[i]));
  }

  // Update C's host data
  HostDataC = OutData;

  // Swap PtrB to be the input
  ParamOut.update(PtrB);
  ParamB.update(PtrC);

  ExecGraph.update(KernelNode);
  Queue.ext_oneapi_graph(ExecGraph).wait();

  // Copy to output data to preserve original data for verifying += op
  Queue.copy(PtrB, OutData.data(), N).wait();
  for (size_t i = 0; i < N; i++) {
    assert(OutData[i] == HostDataB[i] + (HostDataA[i] * HostDataC[i]));
  }
  return 0;
}
