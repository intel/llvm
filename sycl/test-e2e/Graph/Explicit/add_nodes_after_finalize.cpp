// REQUIRES: level_zero, gpu
// RUN: %{build} -o %t.out
// RUN: %{run} %t.out
// RUN: %if ext_oneapi_level_zero %{env ZE_DEBUG=4 %{run} %t.out 2>&1 | FileCheck %s %}
//
// CHECK-NOT: LEAK


// This test adds a new node with the explicit API to an already finalized
// modifiable graph, before finalizing and executing the graph for a second
// time.
// The second run is to check that there are no leaks reported with the embedded
// ZE_DEBUG=4 testing capability.

#include "../graph_common.hpp"

int main() {
  queue Queue;

  using T = unsigned int;

  std::vector<T> DataA(Size), DataB(Size), DataC(Size), DataOut(Size);

  std::iota(DataA.begin(), DataA.end(), 1);
  std::iota(DataB.begin(), DataB.end(), 10);
  std::iota(DataC.begin(), DataC.end(), 1000);
  std::iota(DataOut.begin(), DataOut.end(), 1000);

  std::vector<T> ReferenceC(DataC);
  std::vector<T> ReferenceOut(DataOut);
  for (unsigned n = 0; n < Iterations * 2; n++) {
    for (size_t i = 0; i < Size; i++) {
      ReferenceC[i] += (DataA[i] + DataB[i]);
      if (n >= Iterations)
        ReferenceOut[i] += ReferenceC[i] + 1;
    }
  }

  exp_ext::command_graph Graph{Queue.get_context(), Queue.get_device()};

  T *PtrA = malloc_device<T>(Size, Queue);
  T *PtrB = malloc_device<T>(Size, Queue);
  T *PtrC = malloc_device<T>(Size, Queue);
  T *PtrOut = malloc_device<T>(Size, Queue);

  Queue.copy(DataA.data(), PtrA, Size);
  Queue.copy(DataB.data(), PtrB, Size);
  Queue.copy(DataC.data(), PtrC, Size);
  Queue.copy(DataOut.data(), PtrOut, Size);
  Queue.wait_and_throw();

  auto NodeA = Graph.add([&](handler &CGH) {
    CGH.parallel_for(range<1>(Size),
                     [=](item<1> id) { PtrC[id] += PtrA[id] + PtrB[id]; });
  });

  auto GraphExec = Graph.finalize();

  Graph.add(
      [&](handler &CGH) {
        CGH.parallel_for(range<1>(Size),
                         [=](item<1> id) { PtrOut[id] += PtrC[id] + 1; });
      },
      {exp_ext::property::node::depends_on(NodeA)});

  auto GraphExecAdditional = Graph.finalize();

  event Event;
  for (unsigned n = 0; n < Iterations; n++) {
    Event = Queue.submit([&](handler &CGH) {
      CGH.depends_on(Event);
      CGH.ext_oneapi_graph(GraphExec);
    });
  }

  for (unsigned n = 0; n < Iterations; n++) {
    Event = Queue.submit([&](handler &CGH) {
      CGH.depends_on(Event);
      CGH.ext_oneapi_graph(GraphExecAdditional);
    });
  }

  Queue.wait_and_throw();

  Queue.copy(PtrC, DataC.data(), Size);
  Queue.copy(PtrOut, DataOut.data(), Size);
  Queue.wait_and_throw();

  free(PtrA, Queue);
  free(PtrB, Queue);
  free(PtrC, Queue);
  free(PtrOut, Queue);

  assert(ReferenceC == DataC);
  assert(ReferenceOut == DataOut);

  return 0;
}
