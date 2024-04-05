// This test creates a graph, finalizes it, then submits that as a subgraph of
// another graph, and executes that second graph.

#include "../graph_common.hpp"

int main() {
  queue Queue{};

  using T = short;

  // Values used to modify data inside kernels.
  const int ModValue = 7;
  std::vector<T> DataA(Size), DataB(Size), DataC(Size), DataOut(Size);

  std::iota(DataA.begin(), DataA.end(), 1);
  std::iota(DataB.begin(), DataB.end(), 10);
  std::iota(DataC.begin(), DataC.end(), 1000);
  std::iota(DataOut.begin(), DataOut.end(), 1000);

  // Create reference data for output
  std::vector<T> ReferenceA(DataA);
  std::vector<T> ReferenceB(DataB);
  std::vector<T> ReferenceC(DataC);
  std::vector<T> ReferenceOut(DataOut);
  for (unsigned n = 0; n < Iterations; n++) {
    for (size_t i = 0; i < Size; i++) {
      ReferenceA[i] += ModValue;
      ReferenceB[i] += ModValue;
      ReferenceC[i] = (ReferenceA[i] + ReferenceB[i]);
      ReferenceC[i] -= ModValue;
      ReferenceOut[i] = ReferenceC[i] + ModValue;
    }
  }

  exp_ext::command_graph SubGraph{Queue.get_context(), Queue.get_device()};

  T *PtrA = malloc_device<T>(Size, Queue);
  T *PtrB = malloc_device<T>(Size, Queue);
  T *PtrC = malloc_device<T>(Size, Queue);
  T *PtrOut = malloc_device<T>(Size, Queue);

  Queue.copy(DataA.data(), PtrA, Size);
  Queue.copy(DataB.data(), PtrB, Size);
  Queue.copy(DataC.data(), PtrC, Size);
  Queue.copy(DataOut.data(), PtrOut, Size);
  Queue.wait_and_throw();

  // Add some operations to a graph which will later be submitted as part
  // of another graph.

  // Vector add two values
  auto NodeSubA = add_node(SubGraph, Queue, [&](handler &CGH) {
    CGH.parallel_for(range<1>(Size),
                     [=](item<1> id) { PtrC[id] = PtrA[id] + PtrB[id]; });
  });

  // Modify the output value with some other value
  add_node(
      SubGraph, Queue,
      [&](handler &CGH) {
        depends_on_helper(CGH, NodeSubA);
        CGH.parallel_for(range<1>(Size),
                         [=](item<1> id) { PtrC[id] -= ModValue; });
      },
      NodeSubA);

  auto SubGraphExec = SubGraph.finalize();

  exp_ext::command_graph MainGraph{Queue.get_context(), Queue.get_device()};

  // Modify the input values.
  auto NodeMainA = add_node(MainGraph, Queue, [&](handler &CGH) {
    CGH.parallel_for(range<1>(Size), [=](item<1> id) {
      PtrA[id] += ModValue;
      PtrB[id] += ModValue;
    });
  });

  auto NodeMainB = add_node(
      MainGraph, Queue,
      [&](handler &CGH) {
        depends_on_helper(CGH, NodeMainA);
        CGH.ext_oneapi_graph(SubGraphExec);
      },
      NodeMainA);

  // Copy to another output buffer.
  add_node(
      MainGraph, Queue,
      [&](handler &CGH) {
        depends_on_helper(CGH, NodeMainB);
        CGH.parallel_for(range<1>(Size),
                         [=](item<1> id) { PtrOut[id] = PtrC[id] + ModValue; });
      },
      NodeMainB);

  // Finalize a graph with the additional kernel for writing out to
  auto MainGraphExec = MainGraph.finalize();

  for (unsigned n = 0; n < Iterations; n++) {
    Queue.submit([&](handler &CGH) { CGH.ext_oneapi_graph(MainGraphExec); });
  }
  Queue.wait_and_throw();

  Queue.copy(PtrA, DataA.data(), Size);
  Queue.copy(PtrB, DataB.data(), Size);
  Queue.copy(PtrC, DataC.data(), Size);
  Queue.copy(PtrOut, DataOut.data(), Size);
  Queue.wait_and_throw();

  free(PtrA, Queue);
  free(PtrB, Queue);
  free(PtrC, Queue);
  free(PtrOut, Queue);

  for (size_t i = 0; i < Size; i++) {
    assert(check_value(i, ReferenceA[i], DataA[i], "DataA"));
    assert(check_value(i, ReferenceB[i], DataB[i], "DataB"));
    assert(check_value(i, ReferenceC[i], DataC[i], "DataC"));
    assert(check_value(i, ReferenceOut[i], DataOut[i], "DataOut"));
  }

  return 0;
}
