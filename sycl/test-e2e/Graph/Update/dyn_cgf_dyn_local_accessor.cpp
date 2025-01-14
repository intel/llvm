// RUN: %{build} -o %t.out
// RUN: %{run} %t.out
// Extra run to check for leaks in Level Zero using UR_L0_LEAKS_DEBUG
// RUN: %if level_zero %{env SYCL_PI_LEVEL_ZERO_USE_IMMEDIATE_COMMANDLISTS=0 %{l0_leak_check} %{run} %t.out 2>&1 | FileCheck %s --implicit-check-not=LEAK %}
// Extra run to check for immediate-command-list in Level Zero
// RUN: %if level_zero %{env SYCL_PI_LEVEL_ZERO_USE_IMMEDIATE_COMMANDLISTS=1 %{l0_leak_check} %{run} %t.out 2>&1 | FileCheck %s --implicit-check-not=LEAK %}

// XFAIL: level_zero
// XFAIL-TRACKER: OFNAAO-422

// Tests using dynamic command-group objects with dynamic local accessors.

#include "../graph_common.hpp"

int main() {
  using T = int;

  const size_t LocalMemSize = 128;

  queue Queue{};

  std::vector<T> HostDataBeforeUpdate(Size);
  std::vector<T> HostDataAfterUpdate(Size);
  std::iota(HostDataBeforeUpdate.begin(), HostDataBeforeUpdate.end(), 10);

  T *PtrA = malloc_device<T>(Size, Queue);
  Queue.copy(HostDataBeforeUpdate.data(), PtrA, Size);
  Queue.wait_and_throw();

  exp_ext::command_graph Graph{Queue.get_context(), Queue.get_device()};

  exp_ext::dynamic_local_accessor<T, 1> DynLocalAccessor{Graph, LocalMemSize};

  auto CGFA = [&](handler &CGH) {
    CGH.set_arg(0, DynLocalAccessor);
    auto LocalMem = DynLocalAccessor.get(CGH);

    CGH.parallel_for(nd_range({Size}, {LocalMemSize}), [=](nd_item<1> Item) {
      LocalMem[Item.get_local_linear_id()] = Item.get_local_linear_id();
      PtrA[Item.get_global_linear_id()] = LocalMem[Item.get_local_linear_id()];
    });
  };

  auto CGFB = [&](handler &CGH) {
    CGH.set_arg(0, DynLocalAccessor);
    auto LocalMem = DynLocalAccessor.get(CGH);

    CGH.parallel_for(
        nd_range({Size}, {LocalMemSize * 2}), [=](nd_item<1> Item) {
          LocalMem[Item.get_local_linear_id()] = Item.get_local_linear_id();
          PtrA[Item.get_global_linear_id()] =
              LocalMem[Item.get_local_linear_id()] * 2;
        });
  };

  auto DynamicCG = exp_ext::dynamic_command_group(Graph, {CGFA, CGFB});
  auto DynamicCGNode = Graph.add(DynamicCG);
  auto ExecGraph = Graph.finalize(exp_ext::property::graph::updatable{});

  Queue.submit([&](handler &CGH) { CGH.ext_oneapi_graph(ExecGraph); });
  Queue.wait_and_throw();
  Queue.copy(PtrA, HostDataBeforeUpdate.data(), Size);
  Queue.wait_and_throw();

  DynLocalAccessor.update(LocalMemSize * 2);
  DynamicCG.set_active_index(1);
  ExecGraph.update(DynamicCGNode);
  Queue.ext_oneapi_graph(ExecGraph).wait();

  Queue.submit([&](handler &CGH) { CGH.ext_oneapi_graph(ExecGraph); });
  Queue.wait_and_throw();
  Queue.copy(PtrA, HostDataAfterUpdate.data(), Size);
  Queue.wait_and_throw();

  for (size_t i = 0; i < Size; i++) {
    T Ref = i % LocalMemSize;
    assert(check_value(i, Ref, HostDataBeforeUpdate[i], "PtrA Before Update"));
  }

  for (size_t i = 0; i < Size; i++) {
    T Ref = i % (LocalMemSize * 2) * 2;
    assert(check_value(i, Ref, HostDataAfterUpdate[i], "PtrA After Update"));
  }

  free(PtrA, Queue);

  return 0;
}
