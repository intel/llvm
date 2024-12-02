// RUN: %{build} -o %t.out
// RUN: %{run} %t.out
// Extra run to check for leaks in Level Zero using UR_L0_LEAKS_DEBUG
// RUN: %if level_zero %{env SYCL_PI_LEVEL_ZERO_USE_IMMEDIATE_COMMANDLISTS=0 %{l0_leak_check} %{run} %t.out 2>&1 | FileCheck %s --implicit-check-not=LEAK %}
// Extra run to check for immediate-command-list in Level Zero
// RUN: %if level_zero %{env SYCL_PI_LEVEL_ZERO_USE_IMMEDIATE_COMMANDLISTS=1 %{l0_leak_check} %{run} %t.out 2>&1 | FileCheck %s --implicit-check-not=LEAK %}

// Tests adding a dynamic command-group node to a graph using graph limited
// events for dependencies.

#include "../graph_common.hpp"

int main() {
  queue Queue{};
  exp_ext::command_graph Graph{Queue.get_context(), Queue.get_device()};

  int *PtrA = malloc_device<int>(Size, Queue);
  int *PtrB = malloc_device<int>(Size, Queue);
  int *PtrC = malloc_device<int>(Size, Queue);
  std::vector<int> HostData(Size);

  Graph.begin_recording(Queue);
  int PatternA = 42;
  auto EventA = Queue.fill(PtrA, PatternA, Size);
  int PatternB = 0xA;
  auto EventB = Queue.fill(PtrB, PatternB, Size);
  Graph.end_recording(Queue);

  auto CGFA = [&](handler &CGH) {
    CGH.depends_on({EventA, EventB});
    CGH.parallel_for(Size, [=](item<1> Item) {
      auto I = Item.get_id();
      PtrC[I] = PtrA[I] * PtrB[I];
    });
  };

  auto CGFB = [&](handler &CGH) {
    CGH.depends_on({EventA, EventB});
    CGH.parallel_for(Size, [=](item<1> Item) {
      auto I = Item.get_id();
      PtrC[I] = PtrA[I] + PtrB[I];
    });
  };

  auto DynamicCG = exp_ext::dynamic_command_group(Graph, {CGFA, CGFB});
  auto DynamicCGNode = Graph.add(DynamicCG);
  auto ExecGraph = Graph.finalize(exp_ext::property::graph::updatable{});

  Queue.ext_oneapi_graph(ExecGraph).wait();
  Queue.copy(PtrC, HostData.data(), Size).wait();
  for (size_t i = 0; i < Size; i++) {
    assert(HostData[i] == PatternA * PatternB);
  }

  DynamicCG.set_active_cgf(1);
  ExecGraph.update(DynamicCGNode);

  Queue.ext_oneapi_graph(ExecGraph).wait();
  Queue.copy(PtrC, HostData.data(), Size).wait();
  for (size_t i = 0; i < Size; i++) {
    assert(HostData[i] == PatternA + PatternB);
  }

  sycl::free(PtrA, Queue);
  sycl::free(PtrB, Queue);
  sycl::free(PtrC, Queue);

  return 0;
}
