// REQUIRES: level_zero, gpu
// RUN: %clangxx -fsycl -fsycl-targets=%sycl_triple %s -o %t.out
// RUN: %GPU_RUN_PLACEHOLDER %t.out

// This test creates a graph, finalizes it, then continues to add new nodes to
// the graph with the record & replay API before finalizing and executing the
// second graph.

#include "../graph_common.hpp"

int main() {
  queue TestQueue;

  using T = unsigned int;

  std::vector<T> DataA(Size), DataB(Size), DataC(Size), DataOut(Size);

  std::iota(DataA.begin(), DataA.end(), 1);
  std::iota(DataB.begin(), DataB.end(), 10);
  std::iota(DataC.begin(), DataC.end(), 1000);
  std::iota(DataOut.begin(), DataOut.end(), 1000);

  // Create reference data for output
  std::vector<T> ReferenceC(DataC);
  std::vector<T> ReferenceOut(DataOut);
  for (unsigned n = 0; n < Iterations * 2; n++) {
    for (size_t i = 0; i < Size; i++) {
      ReferenceC[i] += (DataA[i] + DataB[i]);
      if (n >= Iterations)
        ReferenceOut[i] += ReferenceC[i] + 1;
    }
  }

  exp_ext::command_graph Graph{TestQueue.get_context(), TestQueue.get_device()};

  T *PtrA = malloc_device<T>(Size, TestQueue);
  T *PtrB = malloc_device<T>(Size, TestQueue);
  T *PtrC = malloc_device<T>(Size, TestQueue);
  T *PtrOut = malloc_device<T>(Size, TestQueue);

  TestQueue.copy(DataA.data(), PtrA, Size);
  TestQueue.copy(DataB.data(), PtrB, Size);
  TestQueue.copy(DataC.data(), PtrC, Size);
  TestQueue.copy(DataOut.data(), PtrOut, Size);
  TestQueue.wait_and_throw();

  Graph.begin_recording(TestQueue);

  // Vector add to some buffer
  auto Event = TestQueue.submit([&](handler &CGH) {
    CGH.parallel_for(range<1>(Size),
                     [=](item<1> id) { PtrC[id] += PtrA[id] + PtrB[id]; });
  });

  auto GraphExec = Graph.finalize();

  // Read and modify previous output and write to output buffer
  Event = TestQueue.submit([&](handler &CGH) {
    CGH.depends_on(Event);
    CGH.parallel_for(range<1>(Size),
                     [=](item<1> id) { PtrOut[id] += PtrC[id] + 1; });
  });
  Graph.end_recording();

  // Finalize a graph with the additional kernel for writing out to
  auto GraphExecAdditional = Graph.finalize();

  // Execute several iterations of the graph
  for (unsigned n = 0; n < Iterations; n++) {
    Event = TestQueue.submit([&](handler &CGH) {
      CGH.depends_on(Event);
      CGH.ext_oneapi_graph(GraphExec);
    });
  }
  // Execute the extended graph.
  for (unsigned n = 0; n < Iterations; n++) {
    Event = TestQueue.submit([&](handler &CGH) {
      CGH.depends_on(Event);
      CGH.ext_oneapi_graph(GraphExecAdditional);
    });
  }
  TestQueue.wait_and_throw();

  TestQueue.copy(PtrC, DataC.data(), Size);
  TestQueue.copy(PtrOut, DataOut.data(), Size);
  TestQueue.wait_and_throw();

  free(PtrA, TestQueue);
  free(PtrB, TestQueue);
  free(PtrC, TestQueue);
  free(PtrOut, TestQueue);

  assert(ReferenceC == DataC);
  assert(ReferenceOut == DataOut);

  return 0;
}
