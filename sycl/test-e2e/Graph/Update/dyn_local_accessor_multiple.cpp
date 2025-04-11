// RUN: %{build} -o %t.out
// RUN: %{run} %t.out
// Extra run to check for leaks in Level Zero using UR_L0_LEAKS_DEBUG
// RUN: %if level_zero %{env SYCL_PI_LEVEL_ZERO_USE_IMMEDIATE_COMMANDLISTS=0 %{l0_leak_check} %{run} %t.out 2>&1 | FileCheck %s --implicit-check-not=LEAK %}
// Extra run to check for immediate-command-list in Level Zero
// RUN: %if level_zero %{env SYCL_PI_LEVEL_ZERO_USE_IMMEDIATE_COMMANDLISTS=1 %{l0_leak_check} %{run} %t.out 2>&1 | FileCheck %s --implicit-check-not=LEAK %}

// Tests using more than one dynamic_work_group_memory object in the same node.

#include "../graph_common.hpp"
#include <sycl/group_barrier.hpp>

int main() {
  queue Queue{};

  constexpr int LocalSize{16};

  using T = int;

  exp_ext::command_graph Graph{Queue.get_context(), Queue.get_device()};

  int *PtrA = malloc_device<T>(Size, Queue);
  std::vector<T> HostDataA(Size);

  exp_ext::dynamic_local_accessor<T, 1> DynLocalMemA{Graph, LocalSize};
  exp_ext::dynamic_local_accessor<T, 1> DynLocalMemB{Graph, LocalSize};

  Queue.memset(PtrA, 0, Size * sizeof(T)).wait();

  nd_range<1> NDRange{range{Size}, range{LocalSize}};

  auto KernelNode = Graph.add([&](handler &cgh) {
    cgh.parallel_for(NDRange, [=](nd_item<1> Item) {
      size_t GlobalID = Item.get_global_id();
      auto LocalRange = Item.get_local_range(0);

      auto LocalMemA = DynLocalMemA.get();
      auto LocalMemB = DynLocalMemB.get();

      LocalMemA[Item.get_local_id()] = LocalRange;
      LocalMemB[Item.get_local_id()] = LocalRange;
      group_barrier(Item.get_group());

      for (size_t i{0}; i < LocalRange; ++i) {
        PtrA[GlobalID] += (T)(LocalMemA[i] + LocalMemB[i]);
      }
    });
  });

  auto ExecGraph = Graph.finalize(exp_ext::property::graph::updatable{});

  Queue.ext_oneapi_graph(ExecGraph).wait();

  Queue.copy(PtrA, HostDataA.data(), Size).wait();
  for (size_t i = 0; i < Size; i++) {
    assert(HostDataA[i] == LocalSize * LocalSize * 2);
  }

  Queue.memset(PtrA, 0, Size * sizeof(T)).wait();

  constexpr int NewLocalSize{32};

  DynLocalMemA.update(NewLocalSize);
  DynLocalMemB.update(NewLocalSize);
  KernelNode.update_nd_range(nd_range<1>{range{Size}, range{NewLocalSize}});
  ExecGraph.update(KernelNode);

  Queue.ext_oneapi_graph(ExecGraph).wait();

  Queue.copy(PtrA, HostDataA.data(), Size).wait();
  for (size_t i = 0; i < Size; i++) {
    assert(HostDataA[i] == NewLocalSize * NewLocalSize * 2);
  }

  free(PtrA, Queue);
  return 0;
}
