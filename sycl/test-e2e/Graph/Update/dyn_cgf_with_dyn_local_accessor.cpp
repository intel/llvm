// RUN: %{build} -o %t.out
// RUN: %{run} %t.out
// Extra run to check for leaks in Level Zero using UR_L0_LEAKS_DEBUG
// RUN: %if level_zero %{env SYCL_PI_LEVEL_ZERO_USE_IMMEDIATE_COMMANDLISTS=0 %{l0_leak_check} %{run} %t.out 2>&1 | FileCheck %s --implicit-check-not=LEAK %}
// Extra run to check for immediate-command-list in Level Zero
// RUN: %if level_zero %{env SYCL_PI_LEVEL_ZERO_USE_IMMEDIATE_COMMANDLISTS=1 %{l0_leak_check} %{run} %t.out 2>&1 | FileCheck %s --implicit-check-not=LEAK %}

// Tests using a dynamic command-group object with dynamic_local_accessor.

#include "../graph_common.hpp"
#include <sycl/group_barrier.hpp>

int main() {
  queue Queue{};
  exp_ext::command_graph Graph{Queue.get_context(), Queue.get_device()};

  constexpr int LocalSizeA{16};
  constexpr int LocalSizeB{64};

  int *PtrA = malloc_device<int>(Size, Queue);
  int *PtrB = malloc_device<int>(Size, Queue);

  std::vector<int> HostDataA(Size);
  std::vector<int> HostDataB(Size);

  exp_ext::dynamic_local_accessor<int, 1> DynLocalMem(Graph, LocalSizeA);

  nd_range<1> NDrangeA{Size, LocalSizeA};
  auto CGFA = [&](handler &CGH) {
    CGH.parallel_for(NDrangeA, [=](nd_item<1> Item) {
      size_t GlobalID = Item.get_global_id();
      auto LocalRange = Item.get_local_range(0);
      auto LocalMem = DynLocalMem.get();

      LocalMem[Item.get_local_id()] = LocalRange;
      group_barrier(Item.get_group());

      for (size_t i{0}; i < LocalRange; ++i) {
        PtrA[GlobalID] += LocalMem[i];
      }
    });
  };

  nd_range<1> NDrangeB{Size, LocalSizeB};
  auto CGFB = [&](handler &CGH) {
    CGH.parallel_for(NDrangeB, [=](nd_item<1> Item) {
      size_t GlobalID = Item.get_global_id();
      auto LocalRange = Item.get_local_range(0);
      auto LocalMem = DynLocalMem.get();

      LocalMem[Item.get_local_id()] = LocalRange;
      group_barrier(Item.get_group());

      for (size_t i{0}; i < LocalRange; ++i) {
        PtrB[GlobalID] += LocalMem[i];
      }
    });
  };

  auto DynamicCG = exp_ext::dynamic_command_group(Graph, {CGFA, CGFB});
  auto DynamicCGNode = Graph.add(DynamicCG);

  auto ExecGraph = Graph.finalize(exp_ext::property::graph::updatable{});

  auto ExecuteGraphAndVerifyResults = [&](bool A, bool B, const int LocalSize) {
    Queue.memset(PtrA, 0, Size * sizeof(int));
    Queue.memset(PtrB, 0, Size * sizeof(int));
    Queue.wait();

    Queue.ext_oneapi_graph(ExecGraph).wait();

    Queue.copy(PtrA, HostDataA.data(), Size);
    Queue.copy(PtrB, HostDataB.data(), Size);
    Queue.wait();

    for (size_t i = 0; i < Size; i++) {
      assert(HostDataA[i] == (A ? LocalSize * LocalSize : 0));
      assert(HostDataB[i] == (B ? LocalSize * LocalSize : 0));
    }
  };
  ExecuteGraphAndVerifyResults(true, false, LocalSizeA);

  DynamicCG.set_active_index(1);
  DynLocalMem.update(LocalSizeB);
  ExecGraph.update(DynamicCGNode);
  ExecuteGraphAndVerifyResults(false, true, LocalSizeB);

  sycl::free(PtrA, Queue);
  sycl::free(PtrB, Queue);

  return 0;
}
