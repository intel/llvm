// RUN: %{build} -o %t.out
// RUN: %{run} %t.out
// Extra run to check for leaks in Level Zero using UR_L0_LEAKS_DEBUG
// RUN: %if level_zero %{env SYCL_PI_LEVEL_ZERO_USE_IMMEDIATE_COMMANDLISTS=0 %{l0_leak_check} %{run} %t.out 2>&1 | FileCheck %s --implicit-check-not=LEAK %}
// Extra run to check for immediate-command-list in Level Zero
// RUN: %if level_zero %{env SYCL_PI_LEVEL_ZERO_USE_IMMEDIATE_COMMANDLISTS=1 %{l0_leak_check} %{run} %t.out 2>&1 | FileCheck %s --implicit-check-not=LEAK %}

// Tests how the nd-range of a node is overwritten by the active command-group

#include "../graph_common.hpp"

int main() {
  queue Queue{};
  exp_ext::command_graph Graph{Queue.get_context(), Queue.get_device()};

  std::vector<int> HostData(Size);
  int *Ptr = malloc_device<int>(Size, Queue);
  Queue.memset(Ptr, 0, Size * sizeof(int)).wait();

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
  auto DynamicCGNode = Graph.add(DynamicCG);

  size_t NewRange = Size / 2;
  sycl::range<1> UpdateRange(NewRange);
  DynamicCGNode.update_range(UpdateRange);

  DynamicCG.set_active_cgf(1);

  // Check that the UpdateRange from active CGF 0 is preserved
  DynamicCG.set_active_cgf(0);
  auto ExecGraph = Graph.finalize(exp_ext::property::graph::updatable{});

  Queue.ext_oneapi_graph(ExecGraph).wait();
  Queue.copy(Ptr, HostData.data(), Size).wait();
  for (size_t i = 0; i < Size; i++) {
    if (i < NewRange) {
      assert(HostData[i] == PatternA);
    } else {
      assert(HostData[i] == 0);
    }
  }

  sycl::free(Ptr, Queue);
  return 0;
}
