// RUN: %{build} -o %t.out
// RUN: %{run} %t.out
// Extra run to check for leaks in Level Zero using UR_L0_LEAKS_DEBUG
// RUN: %if level_zero %{env SYCL_PI_LEVEL_ZERO_USE_IMMEDIATE_COMMANDLISTS=0 %{l0_leak_check} %{run} %t.out 2>&1 | FileCheck %s --implicit-check-not=LEAK %}
// Extra run to check for immediate-command-list in Level Zero
// RUN: %if level_zero %{env SYCL_PI_LEVEL_ZERO_USE_IMMEDIATE_COMMANDLISTS=1 %{l0_leak_check} %{run} %t.out 2>&1 | FileCheck %s --implicit-check-not=LEAK %}

// Tests updating a dynamic command-group node where the dynamic command-groups
// have different range/nd-range dimensions

#include "../graph_common.hpp"

int main() {
  queue Queue{};
  exp_ext::command_graph Graph{Queue.get_context(), Queue.get_device()};

  const size_t N = 64;
  int *Ptr = malloc_device<int>(N, Queue);
  std::vector<int> HostData(N);

  auto RootNode =
      Graph.add([&](handler &cgh) { cgh.memset(Ptr, 0, N * sizeof(int)); });

  int PatternA = 42;
  sycl::range<1> RangeA{N};
  auto CGFA = [&](handler &CGH) {
    CGH.parallel_for(RangeA,
                     [=](item<1> Item) { Ptr[Item.get_id()] = PatternA; });
  };

  int PatternB = 0xA;
  sycl::nd_range<3> RangeB{sycl::range{4, 4, 4}, sycl::range{2, 2, 2}};
  auto CGFB = [&](handler &CGH) {
    CGH.parallel_for(RangeB, [=](nd_item<3> Item) {
      Ptr[Item.get_global_linear_id()] = PatternB;
    });
  };

  int PatternC = 7;
  sycl::range<2> RangeC{8, 8};
  auto CGFC = [&](handler &CGH) {
    CGH.parallel_for(
        RangeC, [=](item<2> Item) { Ptr[Item.get_linear_id()] = PatternC; });
  };

  auto DynamicCG = exp_ext::dynamic_command_group(Graph, {CGFA, CGFB, CGFC});
  auto DynamicCGNode =
      Graph.add(DynamicCG, exp_ext::property::node::depends_on(RootNode));
  auto ExecGraph = Graph.finalize(exp_ext::property::graph::updatable{});

  Queue.ext_oneapi_graph(ExecGraph).wait();
  Queue.copy(Ptr, HostData.data(), N).wait();
  for (size_t i = 0; i < N; i++) {
    assert(HostData[i] == PatternA);
  }

  DynamicCG.set_active_index(1);
  ExecGraph.update(DynamicCGNode);

  Queue.ext_oneapi_graph(ExecGraph).wait();
  Queue.copy(Ptr, HostData.data(), N).wait();
  for (size_t i = 0; i < N; i++) {
    assert(HostData[i] == PatternB);
  }

  DynamicCG.set_active_index(2);
  ExecGraph.update(DynamicCGNode);

  Queue.ext_oneapi_graph(ExecGraph).wait();
  Queue.copy(Ptr, HostData.data(), N).wait();
  for (size_t i = 0; i < N; i++) {
    assert(HostData[i] == PatternC);
  }

  sycl::free(Ptr, Queue);
  return 0;
}
