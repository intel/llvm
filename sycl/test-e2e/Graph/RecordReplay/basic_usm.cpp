// REQUIRES: level_zero, gpu
// RUN: %{build} -o %t.out
// RUN: %{run} %t.out

// Tests basic queue recording and submission of a graph using USM pointers for
// inputs and outputs.

#include "../graph_common.hpp"

int main() {
  queue Queue;

  using T = int;

  std::vector<T> DataA(Size), DataB(Size), DataC(Size);

  std::iota(DataA.begin(), DataA.end(), 1);
  std::iota(DataB.begin(), DataB.end(), 10);
  std::iota(DataC.begin(), DataC.end(), 1000);

  std::vector<T> ReferenceA(DataA), ReferenceB(DataB), ReferenceC(DataC);
  calculate_reference_data(Iterations, Size, ReferenceA, ReferenceB,
                           ReferenceC);

  {
    ext::oneapi::experimental::command_graph Graph{Queue.get_context(),
                                                   Queue.get_device()};
    auto PtrA = malloc_device<T>(DataA.size(), Queue);
    Queue.memcpy(PtrA, DataA.data(), DataA.size() * sizeof(T)).wait();
    auto PtrB = malloc_device<T>(DataB.size(), Queue);
    Queue.memcpy(PtrB, DataB.data(), DataB.size() * sizeof(T)).wait();
    auto PtrC = malloc_device<T>(DataC.size(), Queue);
    Queue.memcpy(PtrC, DataC.data(), DataC.size() * sizeof(T)).wait();

    Graph.begin_recording(Queue);
    run_kernels_usm(Queue, Size, PtrA, PtrB, PtrC);
    Graph.end_recording();

    auto GraphExec = Graph.finalize();

    event Event;
    for (unsigned n = 0; n < Iterations; n++) {
      Event = Queue.submit([&](handler &CGH) {
        CGH.depends_on(Event);
        CGH.ext_oneapi_graph(GraphExec);
      });
    }
    Queue.wait_and_throw();

    Queue.memcpy(DataA.data(), PtrA, DataA.size() * sizeof(T)).wait();
    Queue.memcpy(DataB.data(), PtrB, DataB.size() * sizeof(T)).wait();
    Queue.memcpy(DataC.data(), PtrC, DataC.size() * sizeof(T)).wait();

    free(PtrA, Queue.get_context());
    free(PtrB, Queue.get_context());
    free(PtrC, Queue.get_context());
  }

  assert(ReferenceA == DataA);
  assert(ReferenceB == DataB);
  assert(ReferenceC == DataC);

  return 0;
}
