// Tests adding a usm memcpy node and submitting the graph.

#include "../graph_common.hpp"

int main() {
  queue Queue{};

  using T = int;

  const T ModValue = 7;
  std::vector<T> DataA(Size), DataB(Size), DataC(Size);

  std::iota(DataA.begin(), DataA.end(), 1);
  std::iota(DataB.begin(), DataB.end(), 10);
  std::iota(DataC.begin(), DataC.end(), 1000);

  // Create reference data for output
  std::vector<T> ReferenceA(DataA), ReferenceB(DataB), ReferenceC(DataC);
  for (size_t i = 0; i < Iterations; i++) {
    for (size_t j = 0; j < Size; j++) {
      ReferenceA[j] = ReferenceB[j];
      ReferenceA[j] += ModValue;
      ReferenceB[j] = ReferenceA[j];
      ReferenceB[j] += ModValue;
      ReferenceC[j] = ReferenceB[j];
    }
  }

  exp_ext::command_graph Graph{Queue.get_context(), Queue.get_device()};

  T *PtrA = malloc_device<T>(Size, Queue);
  T *PtrB = malloc_device<T>(Size, Queue);
  T *PtrC = malloc_device<T>(Size, Queue);

  Queue.copy(DataA.data(), PtrA, Size);
  Queue.copy(DataB.data(), PtrB, Size);
  Queue.copy(DataC.data(), PtrC, Size);
  Queue.wait_and_throw();

  // Copy from B to A
  auto NodeA =
      add_node(Graph, Queue, [&](handler &CGH) { CGH.copy(PtrB, PtrA, Size); });

  // Read & write A
  auto NodeB = add_node(
      Graph, Queue,
      [&](handler &CGH) {
        depends_on_helper(CGH, NodeA);
        CGH.parallel_for(range<1>(Size), [=](item<1> id) {
          auto LinID = id.get_linear_id();
          PtrA[LinID] += ModValue;
        });
      },
      NodeA);

  // Read & write B
  auto NodeModB = add_node(
      Graph, Queue,
      [&](handler &CGH) {
        depends_on_helper(CGH, NodeA);
        CGH.parallel_for(range<1>(Size), [=](item<1> id) {
          auto LinID = id.get_linear_id();
          PtrB[LinID] += ModValue;
        });
      },
      NodeA);

  // memcpy from A to B
  auto NodeC = add_node(
      Graph, Queue,
      [&](handler &CGH) {
        depends_on_helper(CGH, {NodeB, NodeModB});
        CGH.memcpy(PtrB, PtrA, Size * sizeof(T));
      },
      NodeB, NodeModB);

  // Read and write B
  auto NodeD = add_node(
      Graph, Queue,
      [&](handler &CGH) {
        depends_on_helper(CGH, NodeC);
        CGH.parallel_for(range<1>(Size), [=](item<1> id) {
          auto LinID = id.get_linear_id();
          PtrB[LinID] += ModValue;
        });
      },
      NodeC);

  // Copy from B to C
  add_node(
      Graph, Queue,
      [&](handler &CGH) {
        depends_on_helper(CGH, NodeD);
        CGH.copy(PtrB, PtrC, Size);
      },
      NodeD);

  auto GraphExec = Graph.finalize();

  event Event;
  for (unsigned n = 0; n < Iterations; n++) {
    Event =
        Queue.submit([&](handler &CGH) { CGH.ext_oneapi_graph(GraphExec); });
  }

  Queue.copy(PtrA, DataA.data(), Size, Event);
  Queue.copy(PtrB, DataB.data(), Size, Event);
  Queue.copy(PtrC, DataC.data(), Size, Event);
  Queue.wait_and_throw();

  free(PtrA, Queue);
  free(PtrB, Queue);
  free(PtrC, Queue);

  for (size_t i = 0; i < Size; i++) {
    assert(check_value(i, ReferenceA[i], DataA[i], "DataA"));
    assert(check_value(i, ReferenceB[i], DataB[i], "DataB"));
    assert(check_value(i, ReferenceC[i], DataC[i], "DataC"));
  }

  return 0;
}
