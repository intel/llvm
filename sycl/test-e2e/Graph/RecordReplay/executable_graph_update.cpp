// REQUIRES: level_zero, gpu
// RUN: %{build} -o %t.out
// RUN: %{run} %t.out

// Expected fail as executable graph update not implemented yet
// XFAIL: *

// Tests executable graph update by recording two graphs with different buffers
// and attempting to update one from the other.

#include "../graph_common.hpp"

int main() {
  queue Queue;

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

  exp_ext::command_graph GraphA{Queue.get_context(), Queue.get_device()};

  T *PtrA = malloc_device<T>(Size, Queue);
  T *PtrB = malloc_device<T>(Size, Queue);
  T *PtrC = malloc_device<T>(Size, Queue);

  Queue.copy(DataA.data(), PtrA, Size);
  Queue.copy(DataB.data(), PtrB, Size);
  Queue.copy(DataC.data(), PtrC, Size);
  Queue.wait_and_throw();

  GraphA.begin_recording(Queue);
  run_kernels_usm(Queue, Size, PtrA, PtrB, PtrC);
  GraphA.end_recording();

  auto GraphExec = GraphA.finalize();

  exp_ext::command_graph GraphB{Queue.get_context(), Queue.get_device()};

  T *PtrA2 = malloc_device<T>(Size, Queue);
  T *PtrB2 = malloc_device<T>(Size, Queue);
  T *PtrC2 = malloc_device<T>(Size, Queue);

  Queue.copy(DataA2.data(), PtrA2, Size);
  Queue.copy(DataB2.data(), PtrB2, Size);
  Queue.copy(DataC2.data(), PtrC2, Size);
  Queue.wait_and_throw();

  GraphB.begin_recording(Queue);
  run_kernels_usm(Queue, Size, PtrA2, PtrB2, PtrC2);
  GraphB.end_recording();

  event Event;
  // Execute several iterations of the graph for 1st set of buffers
  for (unsigned n = 0; n < Iterations; n++) {
    Event = Queue.submit([&](handler &CGH) {
      CGH.depends_on(Event);
      CGH.ext_oneapi_graph(GraphExec);
    });
  }

  GraphExec.update(GraphB);

  // Execute several iterations of the graph for 2nd set of buffers
  for (unsigned n = 0; n < Iterations; n++) {
    Event = Queue.submit([&](handler &CGH) {
      CGH.depends_on(Event);
      CGH.ext_oneapi_graph(GraphExec);
    });
  }

  Queue.wait_and_throw();

  Queue.copy(PtrA, DataA.data(), Size);
  Queue.copy(PtrB, DataB.data(), Size);
  Queue.copy(PtrC, DataC.data(), Size);

  Queue.copy(PtrA2, DataA2.data(), Size);
  Queue.copy(PtrB2, DataB2.data(), Size);
  Queue.copy(PtrC2, DataC2.data(), Size);
  Queue.wait_and_throw();

  free(PtrA, Queue);
  free(PtrB, Queue);
  free(PtrC, Queue);

  free(PtrA2, Queue);
  free(PtrB2, Queue);
  free(PtrC2, Queue);

  assert(ReferenceA == DataA);
  assert(ReferenceB == DataB);
  assert(ReferenceC == DataC);

  assert(ReferenceA == DataA2);
  assert(ReferenceB == DataB2);
  assert(ReferenceC == DataC2);

  return 0;
}
