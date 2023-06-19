// REQUIRES: level_zero, gpu
// RUN: %{build} -o %t.out
// RUN: %{run} %t.out
// RUN: %if ext_oneapi_level_zero %{env ZE_DEBUG=4 %{run} %t.out 2>&1 | FileCheck %s %}
//
// CHECK-NOT: LEAK

// Tests recording and submission of a graph containing usm memcpy commands.
// The second run is to check that there are no leaks reported with the embedded
// ZE_DEBUG=4 testing capability.

#include "../graph_common.hpp"

int main() {
  queue Queue;

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

  exp_ext::command_graph Graph{Queue.get_context(), Queue.get_device()};

  T *PtrA = malloc_device<T>(Size, Queue);
  T *PtrB = malloc_device<T>(Size, Queue);
  T *PtrC = malloc_device<T>(Size, Queue);

  Queue.copy(DataA.data(), PtrA, Size);
  Queue.copy(DataB.data(), PtrB, Size);
  Queue.copy(DataC.data(), PtrC, Size);
  Queue.wait_and_throw();

  Graph.begin_recording(Queue);

  // memcpy from B to A
  auto EventA = Queue.copy(PtrB, PtrA, Size);

  // Read & write A
  auto EventB = Queue.submit([&](handler &CGH) {
    CGH.depends_on(EventA);
    CGH.parallel_for(range<1>(Size), [=](item<1> id) {
      auto LinID = id.get_linear_id();
      PtrA[LinID] += modValue;
    });
  });

  // memcpy from A to B
  // Test memcpy specifically
  auto EventC = Queue.memcpy(PtrB, PtrA, Size * sizeof(T), EventB);

  // Read and write B
  auto EventD = Queue.submit([&](handler &CGH) {
    CGH.depends_on(EventC);
    CGH.parallel_for(range<1>(Size), [=](item<1> id) {
      auto LinID = id.get_linear_id();
      PtrB[LinID] += modValue;
    });
  });

  // memcpy from B to C
  // Test handler submission instead of shortcut
  Queue.submit([&](handler &CGH) {
    CGH.depends_on(EventD);
    CGH.copy(PtrB, PtrC, Size);
  });

  Graph.end_recording();
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
