// Tests basic adding of nodes with local accessors,
// and submission of the graph.

#include "../graph_common.hpp"

int main() {
  queue Queue{};

  using T = int;

  const size_t LocalSize = 128;

  std::vector<T> DataA(Size), DataB(Size), DataC(Size);

  std::iota(DataA.begin(), DataA.end(), 10);

  std::vector<T> ReferenceA(DataA);

  exp_ext::command_graph Graph{Queue.get_context(), Queue.get_device()};

  T *PtrA = malloc_device<T>(Size, Queue);

  Queue.copy(DataA.data(), PtrA, Size);
  Queue.wait_and_throw();

  auto node = add_node(Graph, Queue, [&](handler &CGH) {
    local_accessor<T, 1> LocalMem(LocalSize, CGH);

    CGH.parallel_for(nd_range({Size}, {LocalSize}), [=](nd_item<1> Item) {
      LocalMem[Item.get_local_linear_id()] = Item.get_global_linear_id() * 2;
      PtrA[Item.get_global_linear_id()] += LocalMem[Item.get_local_linear_id()];
    });
  });

  auto GraphExec = Graph.finalize();

  for (unsigned n = 0; n < Iterations; n++) {
    Queue.submit([&](handler &CGH) { CGH.ext_oneapi_graph(GraphExec); });
  }

  Queue.wait_and_throw();

  Queue.copy(PtrA, DataA.data(), Size);
  Queue.wait_and_throw();

  free(PtrA, Queue);

  for (size_t i = 0; i < Size; i++) {
    T Ref = 10 + i + (i * 2);
    check_value(i, Ref, ReferenceA[i], "PtrA");
  }

  return 0;
}
