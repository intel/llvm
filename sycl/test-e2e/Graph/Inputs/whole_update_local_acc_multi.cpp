// Tests whole graph update of nodes with 2 local accessors,
// and submission of the graph.

#include "../graph_common.hpp"

using T = int;

void add_graph_nodes(
    exp_ext::command_graph<exp_ext::graph_state::modifiable> &Graph,
    queue &Queue, size_t Size, size_t LocalSize, T *Ptr) {
  auto NodeA = add_node(Graph, Queue, [&](handler &CGH) {
    local_accessor<T, 1> LocalMemA(LocalSize, CGH);
    local_accessor<T, 1> LocalMemB(LocalSize, CGH);
    CGH.parallel_for(nd_range({Size}, {LocalSize}), [=](nd_item<1> Item) {
      auto LocalID = Item.get_local_linear_id();
      auto GlobalID = Item.get_global_linear_id();
      LocalMemA[LocalID] = GlobalID;
      LocalMemB[LocalID] = Item.get_local_range(0);
      Ptr[GlobalID] += LocalMemA[LocalID] * LocalMemB[LocalID];
    });
  });

  // Introduce value params so that local arguments are not contiguous indices
  // when set as kernel arguments
  T Constant = 2;
  auto NodeB = add_node(
      Graph, Queue,
      [&](handler &CGH) {
        local_accessor<T, 1> LocalMemA(LocalSize, CGH);
        local_accessor<T, 1> LocalMemB(LocalSize * 2, CGH);

        depends_on_helper(CGH, NodeA);

        CGH.parallel_for(nd_range({Size}, {LocalSize}), [=](nd_item<1> Item) {
          auto LocalID = Item.get_local_linear_id();
          auto GlobalID = Item.get_global_linear_id();
          LocalMemA[LocalID] = GlobalID;
          LocalMemB[LocalID * 2] = Constant;
          LocalMemB[(LocalID * 2) + 1] = Constant;
          Ptr[GlobalID] += LocalMemA[LocalID] * LocalMemB[LocalID * 2] *
                           LocalMemB[(LocalID * 2) + 1];
        });
      },
      NodeA);
}
int main() {
  queue Queue{};

  const size_t LocalSize = 128;

  std::vector<T> DataA(Size), DataB(Size);

  std::iota(DataA.begin(), DataA.end(), 10);
  std::iota(DataB.begin(), DataB.end(), 10);

  exp_ext::command_graph GraphA{Queue.get_context(), Queue.get_device()};

  T *PtrA = malloc_device<T>(Size, Queue);
  T *PtrB = malloc_device<T>(Size, Queue);

  Queue.copy(DataA.data(), PtrA, Size);
  Queue.copy(DataB.data(), PtrB, Size);
  Queue.wait_and_throw();

  size_t GraphALocalSize = LocalSize / 2;
  add_graph_nodes(GraphA, Queue, Size, GraphALocalSize, PtrA);

  auto GraphExecA = GraphA.finalize(exp_ext::property::graph::updatable{});

  // Create second graph for whole graph update with a different local size
  exp_ext::command_graph GraphB{Queue.get_context(), Queue.get_device()};
  add_graph_nodes(GraphB, Queue, Size, LocalSize, PtrB);

  // Execute graphs before updating and check outputs
  for (unsigned n = 0; n < Iterations; n++) {
    Queue.submit([&](handler &CGH) { CGH.ext_oneapi_graph(GraphExecA); });
  }

  Queue.wait_and_throw();

  Queue.copy(PtrA, DataA.data(), Size);
  Queue.copy(PtrB, DataB.data(), Size);
  Queue.wait_and_throw();

  for (size_t i = 0; i < Size; i++) {
    T Init = 10 + i;
    T NodeA = i * GraphALocalSize;
    T NodeB = i * 4;
    T RefA = Init + Iterations * (NodeA + NodeB);
    assert(check_value(i, RefA, DataA[i], "PtrA"));
    assert(check_value(i, Init, DataB[i], "PtrB"));
  }

  // Update GraphExecA using whole graph update
  GraphExecA.update(GraphB);

  // Execute graphs again and check outputs
  for (unsigned n = 0; n < Iterations; n++) {
    Queue.submit([&](handler &CGH) { CGH.ext_oneapi_graph(GraphExecA); });
  }

  Queue.wait_and_throw();

  Queue.copy(PtrA, DataA.data(), Size);
  Queue.copy(PtrB, DataB.data(), Size);
  Queue.wait_and_throw();

  for (size_t i = 0; i < Size; i++) {
    T Init = 10 + i;
    T NodeAGraphA = i * GraphALocalSize;
    T NodeAGraphB = i * LocalSize;
    T NodeB = i * 4;
    T RefA = Init + Iterations * (NodeAGraphA + NodeB);
    T RefB = Init + Iterations * (NodeAGraphB + NodeB);
    assert(check_value(i, RefA, DataA[i], "PtrA"));
    assert(check_value(i, RefB, DataB[i], "PtrB"));
  }

  free(PtrA, Queue);
  free(PtrB, Queue);
  return 0;
}
