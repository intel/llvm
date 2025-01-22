// RUN: %{build} -o %t.out
// RUN: %{run} %t.out
// Extra run to check for leaks in Level Zero using UR_L0_LEAKS_DEBUG
// RUN: %if level_zero %{env SYCL_PI_LEVEL_ZERO_USE_IMMEDIATE_COMMANDLISTS=0 %{l0_leak_check} %{run} %t.out 2>&1 | FileCheck %s --implicit-check-not=LEAK %}
// Extra run to check for immediate-command-list in Level Zero
// RUN: %if level_zero %{env SYCL_PI_LEVEL_ZERO_USE_IMMEDIATE_COMMANDLISTS=1 %{l0_leak_check} %{run} %t.out 2>&1 | FileCheck %s --implicit-check-not=LEAK %}

// Tests updating a dynamic command-group node where the dynamic command-groups
// have different ranges/nd-ranges

#include "../graph_common.hpp"

int main() {
  queue Queue{};
  exp_ext::command_graph Graph{Queue.get_context(), Queue.get_device()};

  int *Ptr = malloc_device<int>(Size, Queue);
  std::vector<int> HostData(Size);

  auto RootNode =
      Graph.add([&](handler &cgh) { cgh.memset(Ptr, 0, Size * sizeof(int)); });

  int PatternA = 42;
  size_t ItemsA = Size / 2;
  sycl::range<1> RangeA{ItemsA};
  auto CGFA = [&](handler &CGH) {
    CGH.parallel_for(RangeA,
                     [=](item<1> Item) { Ptr[Item.get_id()] = PatternA; });
  };

  int PatternB = 0xA;
  size_t ItemsB = Size / 4;
  sycl::nd_range<1> RangeB{sycl::range{ItemsB}, sycl::range{16}};
  auto CGFB = [&](handler &CGH) {
    CGH.parallel_for(
        RangeB, [=](nd_item<1> Item) { Ptr[Item.get_global_id()] = PatternB; });
  };

  auto DynamicCG = exp_ext::dynamic_command_group(Graph, {CGFA, CGFB});
  auto DynamicCGNode =
      Graph.add(DynamicCG, exp_ext::property::node::depends_on(RootNode));
  auto ExecGraph = Graph.finalize(exp_ext::property::graph::updatable{});

  Queue.ext_oneapi_graph(ExecGraph).wait();
  Queue.copy(Ptr, HostData.data(), Size).wait();
  for (size_t i = 0; i < Size; i++) {
    if (i < ItemsA) {
      assert(HostData[i] == PatternA);
    } else {
      assert(HostData[i] == 0);
    }
  }

  DynamicCG.set_active_index(1);
  ExecGraph.update(DynamicCGNode);

  Queue.ext_oneapi_graph(ExecGraph).wait();
  Queue.copy(Ptr, HostData.data(), Size).wait();
  for (size_t i = 0; i < Size; i++) {
    if (i < ItemsB) {
      assert(HostData[i] == PatternB);
    } else {
      assert(HostData[i] == 0);
    }
  }

  sycl::free(Ptr, Queue);
  return 0;
}
