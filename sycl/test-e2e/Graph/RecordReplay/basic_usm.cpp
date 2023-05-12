// REQUIRES: level_zero, gpu
// RUN: %clangxx -fsycl -fsycl-targets=%sycl_triple %s -o %t.out
// RUN: %GPU_RUN_PLACEHOLDER %t.out

// Tests basic queue recording and submission of a graph using USM pointers for
// inputs and outputs.

#include "../graph_common.hpp"

int main() {
  queue TestQueue;

  using T = int;

  std::vector<T> DataA(Size), DataB(Size), DataC(Size);

  std::iota(DataA.begin(), DataA.end(), 1);
  std::iota(DataB.begin(), DataB.end(), 10);
  std::iota(DataC.begin(), DataC.end(), 1000);

  std::vector<T> ReferenceA(DataA), ReferenceB(DataB), ReferenceC(DataC);
  calculate_reference_data(Iterations, Size, ReferenceA, ReferenceB,
                           ReferenceC);

  {
    ext::oneapi::experimental::command_graph Graph{TestQueue.get_context(),
                                                   TestQueue.get_device()};
    auto PtrA = malloc_device<T>(DataA.size(), TestQueue);
    TestQueue.memcpy(PtrA, DataA.data(), DataA.size() * sizeof(T)).wait();
    auto PtrB = malloc_device<T>(DataB.size(), TestQueue);
    TestQueue.memcpy(PtrB, DataB.data(), DataB.size() * sizeof(T)).wait();
    auto PtrC = malloc_device<T>(DataC.size(), TestQueue);
    TestQueue.memcpy(PtrC, DataC.data(), DataC.size() * sizeof(T)).wait();

    Graph.begin_recording(TestQueue);
    run_kernels_usm(TestQueue, Size, PtrA, PtrB, PtrC);
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

    TestQueue.memcpy(DataA.data(), PtrA, DataA.size() * sizeof(T)).wait();
    TestQueue.memcpy(DataB.data(), PtrB, DataB.size() * sizeof(T)).wait();
    TestQueue.memcpy(DataC.data(), PtrC, DataC.size() * sizeof(T)).wait();

    free(PtrA, TestQueue.get_context());
    free(PtrB, TestQueue.get_context());
    free(PtrC, TestQueue.get_context());
  }

  assert(ReferenceA == DataA);
  assert(ReferenceB == DataB);
  assert(ReferenceC == DataC);

  return 0;
}
