// RUN: %{build} -o %t.out
// RUN: %{run} %t.out
// Extra run to check for leaks in Level Zero using UR_L0_LEAKS_DEBUG
// RUN: %if level_zero %{env SYCL_PI_LEVEL_ZERO_USE_IMMEDIATE_COMMANDLISTS=0 %{l0_leak_check} %{run} %t.out 2>&1 | FileCheck %s --implicit-check-not=LEAK %}
// Extra run to check for immediate-command-list in Level Zero
// RUN: %if level_zero %{env SYCL_PI_LEVEL_ZERO_USE_IMMEDIATE_COMMANDLISTS=1 %{l0_leak_check} %{run} %t.out 2>&1 | FileCheck %s --implicit-check-not=LEAK %}

// XFAIL: level_zero
// XFAIL-TRACKER: OFNAAO-307

// Tests updating kernel code using dynamic command-groups that have different
// parameters in each command-group.

#include "../graph_common.hpp"

int main() {
  queue Queue{};
  exp_ext::command_graph Graph{Queue.get_context(), Queue.get_device()};

  int *PtrA = malloc_device<int>(Size, Queue);
  int *PtrB = malloc_device<int>(Size, Queue);

  std::vector<int> HostDataA(Size);
  std::vector<int> HostDataB(Size);

  Queue.memset(PtrA, 0, Size * sizeof(int));
  Queue.memset(PtrB, 0, Size * sizeof(int));
  Queue.wait();

  int PatternA = 0xA;
  auto CGFA = [&](handler &CGH) {
    CGH.parallel_for(Size,
                     [=](item<1> Item) { PtrA[Item.get_id()] = PatternA; });
  };

  int PatternB = 42;
  auto CGFB = [&](handler &CGH) {
    CGH.parallel_for(Size,
                     [=](item<1> Item) { PtrB[Item.get_id()] = PatternB; });
  };

  auto DynamicCG = exp_ext::dynamic_command_group(Graph, {CGFA, CGFB});
  auto DynamicCGNode = Graph.add(DynamicCG);
  auto ExecGraph = Graph.finalize(exp_ext::property::graph::updatable{});

  Queue.ext_oneapi_graph(ExecGraph).wait();
  Queue.copy(PtrA, HostDataA.data(), Size);
  Queue.copy(PtrB, HostDataB.data(), Size);
  Queue.wait();
  for (size_t i = 0; i < Size; i++) {
    assert(HostDataA[i] == PatternA);
    assert(HostDataB[i] == 0);
  }

  DynamicCG.set_active_cgf(1);
  ExecGraph.update(DynamicCGNode);

  Queue.ext_oneapi_graph(ExecGraph).wait();
  Queue.copy(PtrA, HostDataA.data(), Size);
  Queue.copy(PtrB, HostDataB.data(), Size);
  Queue.wait();
  for (size_t i = 0; i < Size; i++) {
    assert(HostDataA[i] == PatternA);
    assert(HostDataB[i] == PatternB);
  }

  sycl::free(PtrA, Queue);
  sycl::free(PtrB, Queue);

  return 0;
}
