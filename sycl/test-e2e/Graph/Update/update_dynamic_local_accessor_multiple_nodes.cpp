// RUN: %{build} -o %t.out
// RUN: %{run} %t.out
// Extra run to check for leaks in Level Zero using UR_L0_LEAKS_DEBUG
// RUN: %if level_zero %{env SYCL_PI_LEVEL_ZERO_USE_IMMEDIATE_COMMANDLISTS=0 %{l0_leak_check} %{run} %t.out 2>&1 | FileCheck %s --implicit-check-not=LEAK %}
// Extra run to check for immediate-command-list in Level Zero
// RUN: %if level_zero %{env SYCL_PI_LEVEL_ZERO_USE_IMMEDIATE_COMMANDLISTS=1 %{l0_leak_check} %{run} %t.out 2>&1 | FileCheck %s --implicit-check-not=LEAK %}

// Tests updating local 2D local accessor parameters in multiple graph nodes
// simultaneously. Also tests using dynamic local accessors with
// non-zero indices.
#include "../graph_common.hpp"

int main() {
  using T = int;

  const size_t LocalMemSize = 128;

  queue Queue{};

  std::vector<T> HostDataBeforeUpdate(Size);
  std::vector<T> HostDataAfterUpdate(Size);
  std::iota(HostDataBeforeUpdate.begin(), HostDataBeforeUpdate.end(), 10);

  T *PtrA = malloc_device<T>(Size, Queue);
  T *PtrB = malloc_device<T>(Size, Queue);
  Queue.copy(HostDataBeforeUpdate.data(), PtrA, Size);
  Queue.copy(HostDataBeforeUpdate.data(), PtrB, Size);
  Queue.wait_and_throw();

  exp_ext::command_graph Graph{Queue.get_context(), Queue.get_device()};

  exp_ext::dynamic_local_accessor<T, 2> DynLocalAccessor{
      Graph, range<2>{LocalMemSize, 2}};

  auto NodeA = Graph.add([&](handler &CGH) {
    CGH.set_arg(1, DynLocalAccessor);
    auto LocalMem = DynLocalAccessor.get(CGH);

    CGH.parallel_for(nd_range({Size}, {LocalMemSize}), [=](nd_item<1> Item) {
      PtrA[Item.get_global_linear_id()] = 0;
      LocalMem[Item.get_local_linear_id()][0] = Item.get_local_linear_id();
      LocalMem[Item.get_local_linear_id()][1] = 2;
      PtrA[Item.get_global_linear_id()] =
          LocalMem[Item.get_local_linear_id()][0] *
          LocalMem[Item.get_local_linear_id()][1];
    });
  });

  auto NodeB = Graph.add(
      [&](handler &CGH) {
        CGH.set_arg(0, DynLocalAccessor);
        auto LocalMem = DynLocalAccessor.get(CGH);

        CGH.parallel_for(nd_range({Size}, {LocalMemSize}),
                         [=](nd_item<1> Item) {
                           LocalMem[Item.get_local_linear_id()][0] =
                               Item.get_local_linear_id();
                           LocalMem[Item.get_local_linear_id()][1] = 4;
                           PtrA[Item.get_global_linear_id()] +=
                               LocalMem[Item.get_local_linear_id()][0] *
                               LocalMem[Item.get_local_linear_id()][1];
                         });
      },
      exp_ext::property::node::depends_on{NodeA});

  auto GraphExec = Graph.finalize(exp_ext::property::graph::updatable{});

  // Submit the graph before the update and save the results.
  Queue.submit([&](handler &CGH) { CGH.ext_oneapi_graph(GraphExec); });
  Queue.wait_and_throw();
  Queue.copy(PtrA, HostDataBeforeUpdate.data(), Size);
  Queue.wait_and_throw();

  DynLocalAccessor.update(range<2>{LocalMemSize * 2, 2});
  NodeA.update_nd_range(nd_range<1>(Size, LocalMemSize * 2));
  NodeB.update_nd_range(nd_range<1>(Size, LocalMemSize * 2));

  GraphExec.update(NodeA);
  GraphExec.update(NodeB);

  // Submit the graph after the update and save the results.
  Queue.submit([&](handler &CGH) { CGH.ext_oneapi_graph(GraphExec); });
  Queue.wait_and_throw();
  Queue.copy(PtrA, HostDataAfterUpdate.data(), Size);
  Queue.wait_and_throw();

  for (size_t i = 0; i < Size; i++) {
    int modI = i % LocalMemSize;
    T Ref = (modI * 2) + (modI * 4);
    assert(check_value(i, Ref, HostDataBeforeUpdate[i], "PtrA Before Update"));
  }

  for (size_t i = 0; i < Size; i++) {
    int modI = i % (LocalMemSize * 2);
    T Ref = (modI * 2) + (modI * 4);
    assert(check_value(i, Ref, HostDataAfterUpdate[i], "PtrA After Update"));
  }

  free(PtrA, Queue);

  return 0;
}
