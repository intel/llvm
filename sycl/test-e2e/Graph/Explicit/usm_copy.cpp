// REQUIRES: level_zero, gpu
// RUN: %{build} -o %t.out
// RUN: %{run} %t.out

// Tests adding a usm memcpy node using the explicit API and submitting
// the graph.

#include "../graph_common.hpp"

int main() {
  queue Queue;

  using T = int;

  const T ModValue = 7;
  std::vector<T> DataA(Size), DataB(Size), DataC(Size);

  std::iota(DataA.begin(), DataA.end(), 1);
  std::iota(DataB.begin(), DataB.end(), 10);
  std::iota(DataC.begin(), DataC.end(), 1000);

  // Create reference data for output
  std::vector<T> ReferenceA(DataA), ReferenceB(DataB), ReferenceC(DataC);
  for (unsigned i = 0; i < Iterations; i++) {
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
  auto NodeA = Graph.add([&](handler &CGH) { CGH.copy(PtrB, PtrA, Size); });

  // Read & write A
  auto NodeB = Graph.add(
      [&](handler &CGH) {
        CGH.parallel_for(range<1>(Size), [=](item<1> id) {
          auto LinID = id.get_linear_id();
          PtrA[LinID] += ModValue;
        });
      },
      {exp_ext::property::node::depends_on(NodeA)});

  // Read & write B
  auto NodeModB = Graph.add(
      [&](handler &CGH) {
        CGH.parallel_for(range<1>(Size), [=](item<1> id) {
          auto LinID = id.get_linear_id();
          PtrB[LinID] += ModValue;
        });
      },
      {exp_ext::property::node::depends_on(NodeA)});

  // memcpy from A to B
  auto NodeC =
      Graph.add([&](handler &CGH) { CGH.memcpy(PtrB, PtrA, Size * sizeof(T)); },
                {exp_ext::property::node::depends_on(NodeB, NodeModB)});

  // Read and write B
  auto NodeD = Graph.add(
      [&](handler &CGH) {
        CGH.parallel_for(range<1>(Size), [=](item<1> id) {
          auto LinID = id.get_linear_id();
          PtrB[LinID] += ModValue;
        });
      },
      {exp_ext::property::node::depends_on(NodeC)});

  // Copy from B to C
  Graph.add([&](handler &CGH) { CGH.copy(PtrB, PtrC, Size); },
            {exp_ext::property::node::depends_on(NodeD)});

  auto GraphExec = Graph.finalize();

  event Event;
  for (unsigned n = 0; n < Iterations; n++) {
    Event = Queue.submit([&](handler &CGH) {
      CGH.depends_on(Event);
      CGH.ext_oneapi_graph(GraphExec);
    });
  }

  Queue.copy(PtrA, DataA.data(), Size, Event);
  Queue.copy(PtrB, DataB.data(), Size, Event);
  Queue.copy(PtrC, DataC.data(), Size, Event);
  Queue.wait_and_throw();

  free(PtrA, Queue);
  free(PtrB, Queue);
  free(PtrC, Queue);

  assert(ReferenceA == DataA);
  assert(ReferenceB == DataB);
  assert(ReferenceC == DataC);

  return 0;
}
