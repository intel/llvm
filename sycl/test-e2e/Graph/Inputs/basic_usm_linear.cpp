// Tests submission of graph that is linear

#include "../graph_common.hpp"

int main() {
  queue Queue{property::queue::in_order()};

  using T = int;

  std::vector<T> DataA(Size), DataB(Size), DataC(Size);

  std::iota(DataA.begin(), DataA.end(), 1);
  std::iota(DataB.begin(), DataB.end(), 10);
  std::iota(DataC.begin(), DataC.end(), 1000);

  std::vector<T> ReferenceA(DataA), ReferenceB(DataB), ReferenceC(DataC);

  exp_ext::command_graph Graph{Queue.get_context(), Queue.get_device()};

  T *PtrA = malloc_device<T>(Size, Queue);
  T *PtrB = malloc_device<T>(Size, Queue);
  T *PtrC = malloc_device<T>(Size, Queue);

  Queue.copy(DataA.data(), PtrA, Size);
  Queue.copy(DataB.data(), PtrB, Size);
  Queue.copy(DataC.data(), PtrC, Size);
  Queue.wait_and_throw();

  auto NodeA = add_node(Graph, Queue, [&](handler &CGH) {
    CGH.parallel_for(range<1>(Size), [=](item<1> id) { PtrC[id] = PtrA[id]; });
  });

  auto NodeB = add_node(
      Graph, Queue,
      [&](handler &CGH) {
        CGH.parallel_for(range<1>(Size),
                         [=](item<1> id) { PtrB[id] += PtrA[id]; });
      },
      NodeA);

  auto NodeC = add_node(
      Graph, Queue,
      [&](handler &CGH) {
        CGH.parallel_for(range<1>(Size),
                         [=](item<1> id) { PtrC[id] += PtrA[id]; });
      },
      NodeB);

  auto GraphExec = Graph.finalize();

  Queue.submit([&](handler &CGH) { CGH.ext_oneapi_graph(GraphExec); });

  Queue.wait_and_throw();

  Queue.copy(PtrA, DataA.data(), Size);
  Queue.copy(PtrB, DataB.data(), Size);
  Queue.copy(PtrC, DataC.data(), Size);
  Queue.wait_and_throw();

  free(PtrA, Queue);
  free(PtrB, Queue);
  free(PtrC, Queue);

  for (size_t i = 0; i < Size; i++) {
    assert(check_value(i, ReferenceA[i], DataA[i], "DataA"));
    assert(check_value(i, ReferenceB[i] + ReferenceA[i], DataB[i], "DataB"));
    assert(check_value(i, ReferenceA[i] + ReferenceA[i], DataC[i], "DataC"));
  }

  Queue.submit([&](handler &CGH) { CGH.ext_oneapi_graph(GraphExec); });

  Queue.copy(PtrA, DataA.data(), Size);
  Queue.copy(PtrB, DataB.data(), Size);
  Queue.copy(PtrC, DataC.data(), Size);
  Queue.wait_and_throw();

  for (size_t i = 0; i < Size; i++) {
    assert(check_value(i, ReferenceA[i], DataA[i], "DataA"));
    assert(check_value(i, ReferenceB[i] + ReferenceA[i] + ReferenceA[i],
                       DataB[i], "DataB"));
    assert(check_value(i, ReferenceA[i] + ReferenceA[i], DataC[i], "DataC"));
  }

  return 0;
}
