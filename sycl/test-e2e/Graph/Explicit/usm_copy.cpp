// REQUIRES: level_zero, gpu
// RUN: %{build} -o %t.out
// RUN: %{run} %t.out

// Expected fail as memcopy not implemented yet
// XFAIL: *

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

  // memcpy from B to A
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

  // memcpy from B to A
  auto NodeC = Graph.add([&](handler &CGH) { CGH.copy(PtrA, PtrB, Size); },
                         {exp_ext::property::node::depends_on(NodeB)});

  // Read and write B
  auto NodeD = Graph.add(
      [&](handler &CGH) {
        CGH.parallel_for(range<1>(Size), [=](item<1> id) {
          auto LinID = id.get_linear_id();
          PtrB[LinID] += ModValue;
        });
      },
      {exp_ext::property::node::depends_on(NodeC)});

  // memcpy from B to C
  Graph.add([&](handler &CGH) { CGH.copy(PtrB, PtrC, Size); },
            {exp_ext::property::node::depends_on(NodeB)});

  auto GraphExec = Graph.finalize();

  event Event;
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

  free(PtrA, Queue);
  free(PtrB, Queue);
  free(PtrC, Queue);

  assert(ReferenceA == DataA);
  assert(ReferenceB == DataB);
  assert(ReferenceC == DataC);

  return 0;
}
