// REQUIRES: level_zero, gpu
// RUN: %clangxx -fsycl -fsycl-targets=%sycl_triple %s -o %t.out
// RUN: %GPU_RUN_PLACEHOLDER %t.out

// Expected fail as host tasks are not implemented yet
// XFAIL: *

// This test uses a host_task within a command_graph recording.

#include "../graph_common.hpp"

int main() {
  queue TestQueue;

  using T = int;

  if (!TestQueue.get_device().has(sycl::aspect::usm_shared_allocations)) {
    return 0;
  }

  const T modValue = T{7};
  std::vector<T> DataA(Size), DataB(Size), DataC(Size);

  std::iota(DataA.begin(), DataA.end(), 1);
  std::iota(DataB.begin(), DataB.end(), 10);
  std::iota(DataC.begin(), DataC.end(), 1000);

  std::vector<T> ReferenceC(DataC);
  for (unsigned n = 0; n < Iterations; n++) {
    for (size_t i = 0; i < Size; i++) {
      ReferenceC[i] += (DataA[i] + DataB[i]) + modValue + 1;
    }
  }

  exp_ext::command_graph Graph{TestQueue.get_context(), TestQueue.get_device()};

  T *PtrA = malloc_device<T>(Size, TestQueue);
  T *PtrB = malloc_device<T>(Size, TestQueue);
  T *PtrC = malloc_shared<T>(Size, TestQueue);

  TestQueue.copy(DataA.data(), PtrA, Size);
  TestQueue.copy(DataB.data(), PtrB, Size);
  TestQueue.copy(DataC.data(), PtrC, Size);
  TestQueue.wait_and_throw();

  Graph.begin_recording(TestQueue);

  // Vector add to output
  event Node1 = TestQueue.submit([&](handler &CGH) {
    CGH.parallel_for(range<1>(Size),
                     [=](item<1> id) { PtrC[id] += PtrA[id] + PtrB[id]; });
  });

  // Modify the output values in a host_task
  auto Node2 = TestQueue.submit([&](handler &CGH) {
    CGH.depends_on(Node1);
    CGH.host_task([=]() {
      for (size_t i = 0; i < Size; i++) {
        PtrC[i] += modValue;
      }
    });
  });

  // Modify temp buffer and write to output buffer
  TestQueue.submit([&](handler &CGH) {
    CGH.depends_on(Node2);
    CGH.parallel_for(range<1>(Size), [=](item<1> id) { PtrC[id] += 1; });
  });
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

  TestQueue.copy(PtrC, DataC.data(), Size);
  TestQueue.wait_and_throw();

  free(PtrA, TestQueue);
  free(PtrB, TestQueue);
  free(PtrC, TestQueue);

  assert(ReferenceC == DataC);

  return 0;
}
