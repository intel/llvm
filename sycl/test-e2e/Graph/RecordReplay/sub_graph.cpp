// REQUIRES: level_zero, gpu
// RUN: %{build} -o %t.out
// RUN: %{run} %t.out
// RUN: %if ext_oneapi_level_zero %{env ZE_DEBUG=4 %{run} %t.out 2>&1 | FileCheck %s %}
//
// CHECK-NOT: LEAK

// This test creates a graph, finalizes it, then submits that as a subgraph of
// another graph using Record & Replay, and executes that second graph.
// The second run is to check that there are no leaks reported with the embedded
// ZE_DEBUG=4 testing capability.

#include "../graph_common.hpp"

int main() {
  queue Queue;

  using T = short;

  // Values used to modify data inside kernels.
  const int ModValue = 7;
  std::vector<T> DataA(Size), DataB(Size), DataC(Size), DataOut(Size);

  std::iota(DataA.begin(), DataA.end(), 1);
  std::iota(DataB.begin(), DataB.end(), 10);
  std::iota(DataC.begin(), DataC.end(), 1000);
  std::iota(DataOut.begin(), DataOut.end(), 1000);

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

  // Record some operations to a graph which will later be submitted as part
  // of another graph.
  SubGraph.begin_recording(Queue);

  // Vector add two values
  auto NodeSubA = Queue.submit([&](handler &CGH) {
    CGH.parallel_for(range<1>(Size),
                     [=](item<1> id) { PtrC[id] = PtrA[id] + PtrB[id]; });
  });

  // Modify the output value with some other value
  Queue.submit([&](handler &CGH) {
    CGH.depends_on(NodeSubA);
    CGH.parallel_for(range<1>(Size), [=](item<1> id) { PtrC[id] -= ModValue; });
  });

  SubGraph.end_recording();

  auto SubGraphExec = SubGraph.finalize();

  exp_ext::command_graph MainGraph{Queue.get_context(), Queue.get_device()};

  MainGraph.begin_recording(Queue);

  // Modify the input values.
  auto NodeMainA = Queue.submit([&](handler &CGH) {
    CGH.parallel_for(range<1>(Size), [=](item<1> id) {
      PtrA[id] += ModValue;
      PtrB[id] += ModValue;
    });
  });

  auto NodeMainB = Queue.submit([&](handler &CGH) {
    CGH.depends_on(NodeMainA);
    CGH.ext_oneapi_graph(SubGraphExec);
  });

  // Copy to another output buffer.
  Queue.submit([&](handler &CGH) {
    CGH.depends_on(NodeMainB);
    CGH.parallel_for(range<1>(Size),
                     [=](item<1> id) { PtrOut[id] = PtrC[id] + ModValue; });
  });

  MainGraph.end_recording();

  // Finalize a graph with the additional kernel for writing out to
  auto MainGraphExec = MainGraph.finalize();

  // Execute several iterations of the graph
  for (unsigned n = 0; n < Iterations; n++) {
    Queue.submit([&](handler &CGH) { CGH.ext_oneapi_graph(MainGraphExec); });
  }
  // Perform a wait on all graph submissions.
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

  assert(ReferenceA == DataA);
  assert(ReferenceB == DataB);
  assert(ReferenceC == DataC);
  assert(ReferenceOut == DataOut);

  return 0;
}
