// Tests executable graph update by creating two graphs with USM ptrs and
// attempting to update one from the other.

#include "../graph_common.hpp"

int main() {
  queue Queue{};

  if (!are_graphs_supported(Queue)) {
    return 0;
  }

  using T = int;

  std::vector<T> DataA(Size), DataB(Size), DataC(Size);

  std::iota(DataA.begin(), DataA.end(), 1);
  std::iota(DataB.begin(), DataB.end(), 10);
  std::iota(DataC.begin(), DataC.end(), 1000);

  auto DataA2 = DataA;
  auto DataB2 = DataB;
  auto DataC2 = DataC;

  std::vector<T> ReferenceA(DataA), ReferenceB(DataB), ReferenceC(DataC);
  calculate_reference_data(Iterations, Size, ReferenceA, ReferenceB,
                           ReferenceC);

  exp_ext::command_graph GraphA{Queue.get_context(), Queue.get_device()};

  T *PtrA = malloc_device<T>(Size, Queue);
  T *PtrB = malloc_device<T>(Size, Queue);
  T *PtrC = malloc_device<T>(Size, Queue);

  Queue.copy(DataA.data(), PtrA, Size);
  Queue.copy(DataB.data(), PtrB, Size);
  Queue.copy(DataC.data(), PtrC, Size);
  Queue.wait_and_throw();

  // Add commands to first graph
  add_nodes(GraphA, Queue, Size, PtrA, PtrB, PtrC);
  auto GraphExec = GraphA.finalize();

  exp_ext::command_graph GraphB{Queue.get_context(), Queue.get_device()};

  T *PtrA2 = malloc_device<T>(Size, Queue);
  T *PtrB2 = malloc_device<T>(Size, Queue);
  T *PtrC2 = malloc_device<T>(Size, Queue);

  Queue.copy(DataA2.data(), PtrA2, Size);
  Queue.copy(DataB2.data(), PtrB2, Size);
  Queue.copy(DataC2.data(), PtrC2, Size);
  Queue.wait_and_throw();

  // Add commands to second graph
  add_nodes(GraphB, Queue, Size, PtrA2, PtrB2, PtrC2);

  // Execute several Iterations of the graph for 1st set of buffers
  event Event;
  for (unsigned n = 0; n < Iterations; n++) {
    Event = Queue.submit([&](handler &CGH) {
      CGH.depends_on(Event);
      CGH.ext_oneapi_graph(GraphExec);
    });
  }

  GraphExec.update(GraphB);

  // Execute several Iterations of the graph for 2nd set of buffers
  for (unsigned n = 0; n < Iterations; n++) {
    Event = Queue.submit([&](handler &CGH) {
      CGH.depends_on(Event);
      CGH.ext_oneapi_graph(GraphExec);
    });
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

    assert(check_value(i, ReferenceA[i], DataA2[i], "DataA2"));
    assert(check_value(i, ReferenceB[i], DataB2[i], "DataB2"));
    assert(check_value(i, ReferenceC[i], DataC2[i], "DataC2"));
  }

  return 0;
}
