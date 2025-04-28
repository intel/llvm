// RUN: %{build} -Wno-error=deprecated-declarations -o %t.out
// RUN: %{run} %t.out
// Extra run to check for leaks in Level Zero using UR_L0_LEAKS_DEBUG
// RUN: %if level_zero %{env SYCL_PI_LEVEL_ZERO_USE_IMMEDIATE_COMMANDLISTS=0 %{l0_leak_check} %{run} %t.out 2>&1 | FileCheck %s --implicit-check-not=LEAK %}
// Extra run to check for immediate-command-list in Level Zero
// RUN: %if level_zero %{env SYCL_PI_LEVEL_ZERO_USE_IMMEDIATE_COMMANDLISTS=1 %{l0_leak_check} %{run} %t.out 2>&1 | FileCheck %s --implicit-check-not=LEAK %}

// Requires constructors that don't take a graph which is currently guarded by
// the breaking changes macro
// REQUIRES: preview-mode

// Tests using dynamic_work_group_memory in a whole graph update graph.

#include "../graph_common.hpp"
#include <sycl/group_barrier.hpp>

int main() {
  queue Queue{};

  using T = int;
  constexpr int LocalSize{16};

  T *Ptr = malloc_device<T>(Size, Queue);
  std::vector<T> HostData(Size);
  std::vector<T> HostOutputCompare(Size, LocalSize * LocalSize);

  Queue.memset(Ptr, 0, Size * sizeof(T)).wait();

  exp_ext::command_graph GraphA{Queue.get_context(), Queue.get_device()};

  exp_ext::dynamic_work_group_memory<T[]> DynLocalMem(LocalSize);

  auto KernelLambda = [=](nd_item<1> Item) {
    size_t GlobalID = Item.get_global_id();
    auto LocalRange = Item.get_local_range(0);

    auto LocalMem = DynLocalMem.get();

    LocalMem[Item.get_local_id()] = LocalRange;
    group_barrier(Item.get_group());

    for (size_t i{0}; i < LocalRange; ++i) {
      Ptr[GlobalID] += LocalMem[i];
    }
  };

  nd_range<1> NDrangeA{Size, LocalSize};
  GraphA.add([&](handler &CGH) { CGH.parallel_for(NDrangeA, KernelLambda); });

  auto GraphExecA = GraphA.finalize();
  Queue.ext_oneapi_graph(GraphExecA).wait();

  Queue.copy(Ptr, HostData.data(), Size).wait();
  for (size_t i = 0; i < Size; i++) {
    assert(check_value(i, HostData[i], HostOutputCompare[i], "HostData"));
  }

  constexpr int NewLocalSize{64};
  DynLocalMem.update(NewLocalSize);
  exp_ext::command_graph GraphB{Queue.get_context(), Queue.get_device()};

  nd_range<1> NDrangeB{Size, NewLocalSize};
  GraphB.add([&](handler &CGH) { CGH.parallel_for(NDrangeB, KernelLambda); });

  auto GraphExecB = GraphB.finalize(exp_ext::property::graph::updatable{});
  GraphExecB.update(GraphA);

  Queue.memset(Ptr, 0, Size * sizeof(T)).wait();
  Queue.ext_oneapi_graph(GraphExecB).wait();

  Queue.copy(Ptr, HostData.data(), Size).wait();
  for (size_t i = 0; i < Size; i++) {
    assert(check_value(i, HostData[i], HostOutputCompare[i], "HostData"));
  }

  return 0;
}
