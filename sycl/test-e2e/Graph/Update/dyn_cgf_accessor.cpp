// RUN: %{build} -o %t.out
// RUN: %{run} %t.out
// Extra run to check for leaks in Level Zero using UR_L0_LEAKS_DEBUG
// RUN: %if level_zero %{env SYCL_PI_LEVEL_ZERO_USE_IMMEDIATE_COMMANDLISTS=0 %{l0_leak_check} %{run} %t.out 2>&1 | FileCheck %s --implicit-check-not=LEAK %}
// Extra run to check for immediate-command-list in Level Zero
// RUN: %if level_zero %{env SYCL_PI_LEVEL_ZERO_USE_IMMEDIATE_COMMANDLISTS=1 %{l0_leak_check} %{run} %t.out 2>&1 | FileCheck %s --implicit-check-not=LEAK %}

// XFAIL: level_zero
// XFAIL-TRACKER: OFNAAO-307

// Tests using dynamic command-group objects with buffer accessors

#include "../graph_common.hpp"

int main() {
  queue Queue{};
  const size_t N = 1024;
  std::vector<int> HostData(N, 0);
  buffer Buf{HostData};
  Buf.set_write_back(false);
  auto Acc = Buf.get_access();

  exp_ext::command_graph Graph{
      Queue.get_context(),
      Queue.get_device(),
      {exp_ext::property::graph::assume_buffer_outlives_graph{}}};

  int PatternA = 42;
  auto CGFA = [&](handler &CGH) {
    CGH.require(Acc);
    CGH.parallel_for(N, [=](item<1> Item) { Acc[Item.get_id()] = PatternA; });
  };

  int PatternB = 0xA;
  auto CGFB = [&](handler &CGH) {
    CGH.require(Acc);
    CGH.parallel_for(N, [=](item<1> Item) { Acc[Item.get_id()] = PatternB; });
  };

  auto DynamicCG = exp_ext::dynamic_command_group(Graph, {CGFA, CGFB});
  auto DynamicCGNode = Graph.add(DynamicCG);
  auto ExecGraph = Graph.finalize(exp_ext::property::graph::updatable{});

  Queue.ext_oneapi_graph(ExecGraph).wait();
  Queue.copy(Acc, HostData.data()).wait();
  for (size_t i = 0; i < N; i++) {
    assert(HostData[i] == PatternA);
  }

  DynamicCG.set_active_cgf(1);
  ExecGraph.update(DynamicCGNode);
  Queue.ext_oneapi_graph(ExecGraph).wait();

  Queue.copy(Acc, HostData.data()).wait();
  for (size_t i = 0; i < N; i++) {
    assert(HostData[i] == PatternB);
  }

  return 0;
}
