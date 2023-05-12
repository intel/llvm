// REQUIRES: level_zero, gpu
// RUN: %clangxx -fsycl -fsycl-targets=%sycl_triple %s -o %t.out
// RUN: %GPU_RUN_PLACEHOLDER %t.out

// Expected fail as whole graph update not implemented yet
// XFAIL: *

// Tests executable graph update by creating two graphs with USM ptrs and
// attempting to update one from the other.

#include "../graph_common.hpp"

int main() {
  queue TestQueue;

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

  exp_ext::command_graph GraphA{TestQueue.get_context(),
                                TestQueue.get_device()};

  T *PtrA = malloc_device<T>(Size, TestQueue);
  T *PtrB = malloc_device<T>(Size, TestQueue);
  T *PtrC = malloc_device<T>(Size, TestQueue);

  TestQueue.copy(DataA.data(), PtrA, Size);
  TestQueue.copy(DataB.data(), PtrB, Size);
  TestQueue.copy(DataC.data(), PtrC, Size);
  TestQueue.wait_and_throw();

  // Add commands to first graph
  add_kernels_usm(GraphA, Size, PtrA, PtrB, PtrC);
  auto GraphExec = GraphA.finalize();

  exp_ext::command_graph GraphB{TestQueue.get_context(),
                                TestQueue.get_device()};

  T *PtrA2 = malloc_device<T>(Size, TestQueue);
  T *PtrB2 = malloc_device<T>(Size, TestQueue);
  T *PtrC2 = malloc_device<T>(Size, TestQueue);

  TestQueue.copy(DataA2.data(), PtrA2, Size);
  TestQueue.copy(DataB2.data(), PtrB2, Size);
  TestQueue.copy(DataC2.data(), PtrC2, Size);
  TestQueue.wait_and_throw();

  // Add commands to second graph
  add_kernels_usm(GraphB, Size, PtrA2, PtrB2, PtrC2);

  // Execute several Iterations of the graph for 1st set of buffers
  event Event;
  for (unsigned n = 0; n < Iterations; n++) {
    Event = TestQueue.submit([&](handler &CGH) {
      CGH.depends_on(Event);
      CGH.ext_oneapi_graph(GraphExec);
    });
  }

  GraphExec.update(GraphB);

  // Execute several Iterations of the graph for 2nd set of buffers
  for (unsigned n = 0; n < Iterations; n++) {
    Event = TestQueue.submit([&](handler &CGH) {
      CGH.depends_on(Event);
      CGH.ext_oneapi_graph(GraphExec);
    });
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

  assert(ReferenceA == DataA2);
  assert(ReferenceB == DataB2);
  assert(ReferenceC == DataC2);

  return 0;
}
