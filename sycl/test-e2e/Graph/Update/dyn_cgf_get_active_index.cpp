// RUN: %{build} -o %t.out
// RUN: %{run} %t.out
// Extra run to check for leaks in Level Zero using UR_L0_LEAKS_DEBUG
// RUN: %if level_zero %{env SYCL_PI_LEVEL_ZERO_USE_IMMEDIATE_COMMANDLISTS=0 %{l0_leak_check} %{run} %t.out 2>&1 | FileCheck %s --implicit-check-not=LEAK %}
// Extra run to check for immediate-command-list in Level Zero
// RUN: %if level_zero %{env SYCL_PI_LEVEL_ZERO_USE_IMMEDIATE_COMMANDLISTS=1 %{l0_leak_check} %{run} %t.out 2>&1 | FileCheck %s --implicit-check-not=LEAK %}

// Tests the `get_active_index()` query

#include "../graph_common.hpp"

int main() {
  queue Queue{};
  exp_ext::command_graph Graph{Queue.get_context(), Queue.get_device()};

  int *Ptr = malloc_device<int>(Size, Queue);
  std::vector<int> HostData(Size);

  int PatternA = 42;
  auto CGFA = [&](handler &CGH) {
    CGH.parallel_for(Size,
                     [=](item<1> Item) { Ptr[Item.get_id()] = PatternA; });
  };

  int PatternB = 0xA;
  auto CGFB = [&](handler &CGH) {
    CGH.parallel_for(Size,
                     [=](item<1> Item) { Ptr[Item.get_id()] = PatternB; });
  };

  auto DynamicCG = exp_ext::dynamic_command_group(Graph, {CGFA, CGFB});
  size_t ActiveIndex = DynamicCG.get_active_index();
  assert(0 == ActiveIndex); // Active index is zero by default

  // Set active index to 1 before adding node to graph
  DynamicCG.set_active_index(1);
  ActiveIndex = DynamicCG.get_active_index();
  assert(1 == ActiveIndex);

  auto DynamicCGNode = Graph.add(DynamicCG);

  // Set active index to 0 before finalizing the graph
  DynamicCG.set_active_index(0);
  ActiveIndex = DynamicCG.get_active_index();
  assert(0 == ActiveIndex);
  auto ExecGraph = Graph.finalize(exp_ext::property::graph::updatable{});

  Queue.ext_oneapi_graph(ExecGraph).wait();
  Queue.copy(Ptr, HostData.data(), Size).wait();
  for (size_t i = 0; i < Size; i++) {
    assert(HostData[i] == PatternA);
  }

  // Set active index to 1 before updating the graph
  DynamicCG.set_active_index(1);
  ActiveIndex = DynamicCG.get_active_index();
  assert(1 == ActiveIndex);

  ExecGraph.update(DynamicCGNode);

  Queue.ext_oneapi_graph(ExecGraph).wait();
  Queue.copy(Ptr, HostData.data(), Size).wait();
  for (size_t i = 0; i < Size; i++) {
    assert(HostData[i] == PatternB);
  }

  sycl::free(Ptr, Queue);

  return 0;
}
