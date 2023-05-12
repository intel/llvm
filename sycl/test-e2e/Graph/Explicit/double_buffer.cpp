// REQUIRES: level_zero, gpu
// RUN: %clangxx -fsycl -fsycl-targets=%sycl_triple %s -o %t.out
// RUN: %GPU_RUN_PLACEHOLDER %t.out

// Expected fail as executable graph update isn't implemented yet
// XFAIL: *

// Tests executable graph update by creating a double buffering scenario, where
// a single graph is repeatedly executed then updated to swap between two sets
// of buffers.

#include "../graph_common.hpp"

int main() {
  queue TestQueue;

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

  exp_ext::command_graph Graph{TestQueue.get_context(), TestQueue.get_device()};

  T *PtrA = malloc_device<T>(Size, TestQueue);
  T *PtrB = malloc_device<T>(Size, TestQueue);
  T *PtrC = malloc_device<T>(Size, TestQueue);

  T *PtrA2 = malloc_device<T>(Size, TestQueue);
  T *PtrB2 = malloc_device<T>(Size, TestQueue);
  T *PtrC2 = malloc_device<T>(Size, TestQueue);

  TestQueue.copy(DataA.data(), PtrA, Size);
  TestQueue.copy(DataB.data(), PtrB, Size);
  TestQueue.copy(DataC.data(), PtrC, Size);

  TestQueue.copy(DataA2.data(), PtrA, Size);
  TestQueue.copy(DataB2.data(), PtrB, Size);
  TestQueue.copy(DataC2.data(), PtrC, Size);
  TestQueue.wait_and_throw();

  add_kernels_usm(Graph, Size, PtrA, PtrB, PtrC);

  auto ExecGraph = Graph.finalize();

  // Create second graph using other buffer set
  exp_ext::command_graph GraphUpdate{TestQueue.get_context(),
                                     TestQueue.get_device()};
  add_kernels_usm(GraphUpdate, Size, PtrA, PtrB, PtrC);

  event Event;
  for (unsigned i = 0; i < Iterations; i++) {
    Event = TestQueue.submit([&](handler &CGH) {
      CGH.depends_on(Event);
      CGH.ext_oneapi_graph(ExecGraph);
    });
    // Update to second set of buffers
    ExecGraph.update(GraphUpdate);
    Event = TestQueue.submit([&](handler &CGH) {
      CGH.depends_on(Event);
      CGH.ext_oneapi_graph(ExecGraph);
    });
    // Reset back to original buffers
    ExecGraph.update(Graph);
  }

  TestQueue.wait_and_throw();

  TestQueue.copy(PtrA, DataA.data(), Size);
  TestQueue.copy(PtrB, DataB.data(), Size);
  TestQueue.copy(PtrC, DataC.data(), Size);

  TestQueue.copy(PtrA2, DataA2.data(), Size);
  TestQueue.copy(PtrB2, DataB2.data(), Size);
  TestQueue.copy(PtrC2, DataC2.data(), Size);
  TestQueue.wait_and_throw();

  free(PtrA, TestQueue);
  free(PtrB, TestQueue);
  free(PtrC, TestQueue);

  free(PtrA2, TestQueue);
  free(PtrB2, TestQueue);
  free(PtrC2, TestQueue);

  assert(ReferenceA == DataA);
  assert(ReferenceB == DataB);
  assert(ReferenceC == DataC);

  assert(ReferenceA2 == DataA2);
  assert(ReferenceB2 == DataB2);
  assert(ReferenceC2 == DataC2);

  return 0;
}
