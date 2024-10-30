// Tests whole graph update of nodes with local accessors,
// and submission of the graph.

#include "../graph_common.hpp"

using T = int;

auto add_graph_node(
    exp_ext::command_graph<exp_ext::graph_state::modifiable> &Graph,
    queue &Queue, size_t Size, size_t LocalSize, T *Ptr) {
  return add_node(Graph, Queue, [&](handler &CGH) {
    local_accessor<T, 1> LocalMem(LocalSize, CGH);

    CGH.parallel_for(nd_range({Size}, {LocalSize}), [=](nd_item<1> Item) {
      LocalMem[Item.get_local_linear_id()] = Item.get_global_linear_id() * 2;
      Ptr[Item.get_global_linear_id()] +=
          LocalMem[Item.get_local_linear_id()] + Item.get_local_range(0);
    });
  });
}
int main() {
  queue Queue{};

  const size_t LocalSize = 128;

  std::vector<T> DataA(Size), DataB(Size);

  std::iota(DataA.begin(), DataA.end(), 10);
  std::iota(DataB.begin(), DataB.end(), 10);

  std::vector<T> ReferenceA(DataA), ReferenceB(DataB);

  exp_ext::command_graph GraphA{Queue.get_context(), Queue.get_device()};

  T *PtrA = malloc_device<T>(Size, Queue);
  T *PtrB = malloc_device<T>(Size, Queue);

  Queue.copy(DataA.data(), PtrA, Size);
  Queue.copy(DataB.data(), PtrB, Size);
  Queue.wait_and_throw();

  auto NodeA = add_graph_node(GraphA, Queue, Size, LocalSize / 2, PtrA);

  auto GraphExecA = GraphA.finalize(exp_ext::property::graph::updatable{});

  // Create second graph for whole graph update with a different local size
  exp_ext::command_graph GraphB{Queue.get_context(), Queue.get_device()};
  auto NodeB = add_graph_node(GraphB, Queue, Size, LocalSize, PtrB);

  // Execute graphs before updating and check outputs
  for (unsigned n = 0; n < Iterations; n++) {
    Queue.submit([&](handler &CGH) { CGH.ext_oneapi_graph(GraphExecA); });
  }

  Queue.wait_and_throw();

  Queue.copy(PtrA, DataA.data(), Size);
  Queue.copy(PtrB, DataB.data(), Size);
  Queue.wait_and_throw();

  for (size_t i = 0; i < Size; i++) {
    T RefA = 10 + i + (i * 2) + LocalSize / 2;
    T RefB = 10 + i;
    check_value(i, RefA, ReferenceA[i], "PtrA");
    check_value(i, RefB, ReferenceB[i], "PtrB");
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
    T RefA = 10 + i + (i * 2) + LocalSize / 2;
    T RefB = 10 + i + (i * 2) + LocalSize;
    check_value(i, RefA, ReferenceA[i], "PtrA");
    check_value(i, RefB, ReferenceB[i], "PtrB");
  }

  free(PtrA, Queue);
  free(PtrB, Queue);
  return 0;
}
