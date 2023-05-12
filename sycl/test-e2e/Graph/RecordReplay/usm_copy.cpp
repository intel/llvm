// REQUIRES: level_zero, gpu
// RUN: %clangxx -fsycl -fsycl-targets=%sycl_triple %s -o %t.out
// RUN: %GPU_RUN_PLACEHOLDER %t.out

// Expected fail as mem copy not implemented yet
// XFAIL: *

// Tests recording and submission of a graph containing usm memcpy commands.

#include "../graph_common.hpp"

int main() {
  queue TestQueue;

  using T = int;

  const T modValue = 7;
  std::vector<T> DataA(Size), DataB(Size), DataC(Size);

  std::iota(DataA.begin(), DataA.end(), 1);
  std::iota(DataB.begin(), DataB.end(), 10);
  std::iota(DataC.begin(), DataC.end(), 1000);

  std::vector<T> ReferenceA(DataA), ReferenceB(DataB), ReferenceC(DataC);
  for (size_t i = 0; i < Iterations; i++) {
    for (size_t j = 0; j < Size; j++) {
      ReferenceA[j] = ReferenceB[j];
      ReferenceA[j] += modValue;
      ReferenceB[j] = ReferenceA[j];
      ReferenceB[j] += modValue;
      ReferenceC[j] = ReferenceB[j];
    }
  }

  exp_ext::command_graph Graph{TestQueue.get_context(), TestQueue.get_device()};

  T *PtrA = malloc_device<T>(Size, TestQueue);
  T *PtrB = malloc_device<T>(Size, TestQueue);
  T *PtrC = malloc_device<T>(Size, TestQueue);

  TestQueue.copy(DataA.data(), PtrA, Size);
  TestQueue.copy(DataB.data(), PtrB, Size);
  TestQueue.copy(DataC.data(), PtrC, Size);
  TestQueue.wait_and_throw();

  Graph.begin_recording(TestQueue);

  // memcpy from B to A
  auto EventA = TestQueue.copy(PtrB, PtrA, Size);

  // Read & write A
  auto EventB = TestQueue.submit([&](handler &CGH) {
    CGH.depends_on(EventA);
    CGH.parallel_for(range<1>(Size), [=](item<1> id) {
      auto LinID = id.get_linear_id();
      PtrA[LinID] += modValue;
    });
  });

  // memcpy from A to B
  auto EventC = TestQueue.copy(PtrA, PtrB, Size, EventB);

  // Read and write B
  auto EventD = TestQueue.submit([&](handler &CGH) {
    CGH.depends_on(EventC);
    CGH.parallel_for(range<1>(Size), [=](item<1> id) {
      auto LinID = id.get_linear_id();
      PtrB[LinID] += modValue;
    });
  });

  // memcpy from B to C
  TestQueue.copy(PtrB, PtrC, Size, EventD);

  Graph.end_recording();
  auto GraphExec = Graph.finalize();

  event Event;
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

  free(PtrA, TestQueue);
  free(PtrB, TestQueue);
  free(PtrC, TestQueue);

  assert(ReferenceA == DataA);
  assert(ReferenceB == DataB);
  assert(ReferenceC == DataC);

  return 0;
}
