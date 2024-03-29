// RUN: %{build} -o %t.out
// RUN: %{run} %t.out
// Extra run to check for leaks in Level Zero using UR_L0_LEAKS_DEBUG
// RUN: %if level_zero %{env SYCL_PI_LEVEL_ZERO_USE_IMMEDIATE_COMMANDLISTS=0 %{l0_leak_check} %{run} %t.out 2>&1 | FileCheck %s --implicit-check-not=LEAK %}
// Extra run to check for immediate-command-list in Level Zero
// RUN: %if level_zero && linux %{env SYCL_PI_LEVEL_ZERO_USE_IMMEDIATE_COMMANDLISTS=1 %{l0_leak_check} %{run} %t.out 2>&1 | FileCheck %s --implicit-check-not=LEAK %}
//
// UNSUPPORTED: opencl, level_zero

// Tests updating a graph node using index-based explicit update

#include "../graph_common.hpp"

int main() {
  queue Queue{};

  const size_t N = 1024;

  exp_ext::command_graph Graph{Queue.get_context(), Queue.get_device()};

  int *PtrA = malloc_device<int>(N, Queue);

  std::vector<int> HostDataA(N);

  Queue.memset(PtrA, 0, N * sizeof(int)).wait();

  range<1> Range{1024};

  auto KernelNode = Graph.add([&](handler &cgh) {
    cgh.parallel_for(Range, [=](item<1> Item) {
      size_t GlobalID = Item.get_id();
      PtrA[GlobalID] += GlobalID;
    });
  });

  auto ExecGraph = Graph.finalize(exp_ext::property::graph::updatable{});

  // first half of PtrA should be filled with values
  Queue.ext_oneapi_graph(ExecGraph).wait();

  Queue.copy(PtrA, HostDataA.data(), N).wait();
  for (size_t i = 0; i < N; i++) {
    assert(HostDataA[i] == i);
  }

  // Update NDRange to target first half only
  KernelNode.update_range(range<1>{512});
  ExecGraph.update(KernelNode);
  Queue.ext_oneapi_graph(ExecGraph).wait();

  Queue.copy(PtrA, HostDataA.data(), N).wait();
  for (size_t i = 0; i < N; i++) {
    assert(HostDataA[i] == (i >= 512 ? i : i * 2));
  }
  return 0;
}
