// RUN: %{build} -o %t.out
// RUN: %{run} %t.out
// Extra run to check for leaks in Level Zero using UR_L0_LEAKS_DEBUG
// RUN: %if level_zero %{env SYCL_PI_LEVEL_ZERO_USE_IMMEDIATE_COMMANDLISTS=0 %{l0_leak_check} %{run} %t.out 2>&1 | FileCheck %s --implicit-check-not=LEAK %}
// Extra run to check for immediate-command-list in Level Zero
// RUN: %if level_zero %{env SYCL_PI_LEVEL_ZERO_USE_IMMEDIATE_COMMANDLISTS=1 %{l0_leak_check} %{run} %t.out 2>&1 | FileCheck %s --implicit-check-not=LEAK %}

// Tests using the same dynamic command-group in more than one graph node.

#include "../graph_common.hpp"

int main() {
  queue Queue{};
  exp_ext::command_graph Graph{Queue.get_context(), Queue.get_device()};

  std::vector<int> HostData(Size);
  int *Ptr = malloc_device<int>(Size, Queue);

  auto RootNode =
      Graph.add([&](handler &CGH) { CGH.memset(Ptr, 0, Size * sizeof(int)); });

  int PatternA = 42;
  auto CGFA = [&](handler &CGH) {
    CGH.parallel_for(Size,
                     [=](item<1> Item) { Ptr[Item.get_id()] += PatternA; });
  };

  int PatternB = 0xA;
  auto CGFB = [&](handler &CGH) {
    CGH.parallel_for(Size,
                     [=](item<1> Item) { Ptr[Item.get_id()] += PatternB; });
  };

  auto DynamicCG = exp_ext::dynamic_command_group(Graph, {CGFA, CGFB});
  auto Node1 =
      Graph.add(DynamicCG, exp_ext::property::node::depends_on(RootNode));

  auto Node2 = Graph.add(
      [&](handler &cgh) {
        cgh.parallel_for(Size, [=](item<1> Item) { Ptr[Item.get_id()] *= 2; });
      },
      exp_ext::property::node::depends_on(Node1));

  auto Node3 = Graph.add(DynamicCG, exp_ext::property::node::depends_on(Node2));

  // This ND-Range affects Node 1 as well, as the range is tied to the node.
  sycl::range<1> Node3Range(Size / 2);
  Node3.update_range(Node3Range);
  auto ExecGraph = Graph.finalize(exp_ext::property::graph::updatable{});

  Queue.ext_oneapi_graph(ExecGraph).wait();
  Queue.copy(Ptr, HostData.data(), Size).wait();
  for (size_t i = 0; i < Size; i++) {
    int Ref = (i < Node3Range.get(0)) ? (PatternA * 3) : 0;
    assert(HostData[i] == Ref);
  }

  DynamicCG.set_active_index(1);
  ExecGraph.update(Node1);
  ExecGraph.update(Node3);

  Queue.ext_oneapi_graph(ExecGraph).wait();
  Queue.copy(Ptr, HostData.data(), Size).wait();
  int Ref = (PatternB * 3);
  for (size_t i = 0; i < Size; i++) {
    assert(HostData[i] == Ref);
  }

  sycl::free(Ptr, Queue);

  return 0;
}
