// Tests executable graph update by creating a double buffering scenario, where
// a single graph is repeatedly executed then updated to swap between two sets
// of buffers.

#include "../graph_common.hpp"

int main() {
  queue Queue{};

  using T = int;

  std::vector<T> DataA(Size), DataB(Size), DataC(Size);
  std::vector<T> DataA2(Size), DataB2(Size), DataC2(Size);

  std::iota(DataA.begin(), DataA.end(), 1);
  std::iota(DataB.begin(), DataB.end(), 10);
  std::iota(DataC.begin(), DataC.end(), 1000);

  std::iota(DataA2.begin(), DataA2.end(), 3);
  std::iota(DataB2.begin(), DataB2.end(), 13);
  std::iota(DataC2.begin(), DataC2.end(), 1333);

  std::vector<T> ReferenceA(DataA), ReferenceB(DataB), ReferenceC(DataC);
  std::vector<T> ReferenceA2(DataA2), ReferenceB2(DataB2), ReferenceC2(DataC2);

  calculate_reference_data(Iterations, Size, ReferenceA, ReferenceB,
                           ReferenceC);
  calculate_reference_data(Iterations, Size, ReferenceA2, ReferenceB2,
                           ReferenceC2);

  exp_ext::command_graph Graph{Queue.get_context(), Queue.get_device()};

  T *PtrA = malloc_device<T>(Size, Queue);
  T *PtrB = malloc_device<T>(Size, Queue);
  T *PtrC = malloc_device<T>(Size, Queue);

  T *PtrA2 = malloc_device<T>(Size, Queue);
  T *PtrB2 = malloc_device<T>(Size, Queue);
  T *PtrC2 = malloc_device<T>(Size, Queue);

  Queue.copy(DataA.data(), PtrA, Size);
  Queue.copy(DataB.data(), PtrB, Size);
  Queue.copy(DataC.data(), PtrC, Size);

  Queue.copy(DataA2.data(), PtrA, Size);
  Queue.copy(DataB2.data(), PtrB, Size);
  Queue.copy(DataC2.data(), PtrC, Size);
  Queue.wait_and_throw();

  add_nodes(Graph, Queue, Size, PtrA, PtrB, PtrC);

  auto ExecGraph = Graph.finalize();

  // Create second graph using other buffer set
  exp_ext::command_graph GraphUpdate{Queue.get_context(), Queue.get_device()};
  add_nodes(GraphUpdate, Queue, Size, PtrA, PtrB, PtrC);

  event Event;
  for (size_t i = 0; i < Iterations; i++) {
    Event = Queue.submit([&](handler &CGH) {
      CGH.depends_on(Event);
      CGH.ext_oneapi_graph(ExecGraph);
    });
    // Update to second set of buffers
    ExecGraph.update(GraphUpdate);
    Event = Queue.submit([&](handler &CGH) {
      CGH.depends_on(Event);
      CGH.ext_oneapi_graph(ExecGraph);
    });
    // Reset back to original buffers
    ExecGraph.update(Graph);
  }

  Queue.wait_and_throw();

  Queue.copy(PtrA, DataA.data(), Size);
  Queue.copy(PtrB, DataB.data(), Size);
  Queue.copy(PtrC, DataC.data(), Size);

  Queue.copy(PtrA2, DataA2.data(), Size);
  Queue.copy(PtrB2, DataB2.data(), Size);
  Queue.copy(PtrC2, DataC2.data(), Size);
  Queue.wait_and_throw();

  free(PtrA, Queue);
  free(PtrB, Queue);
  free(PtrC, Queue);

  free(PtrA2, Queue);
  free(PtrB2, Queue);
  free(PtrC2, Queue);

  for (size_t i = 0; i < Size; i++) {
    assert(check_value(i, ReferenceA[i], DataA[i], "DataA"));
    assert(check_value(i, ReferenceB[i], DataB[i], "DataB"));
    assert(check_value(i, ReferenceC[i], DataC[i], "DataC"));

    assert(check_value(i, ReferenceA2[i], DataA2[i], "DataA2"));
    assert(check_value(i, ReferenceB2[i], DataB2[i], "DataB2"));
    assert(check_value(i, ReferenceC2[i], DataC2[i], "DataC2"));
  }

  return 0;
}
