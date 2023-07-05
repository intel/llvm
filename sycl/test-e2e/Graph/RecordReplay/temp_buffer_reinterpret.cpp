// REQUIRES: level_zero, gpu
// RUN: %{build} -o %t.out
// RUN: %{run} %t.out
// RUN: %if ext_oneapi_level_zero %{env ZE_DEBUG=4 %{run} %t.out 2>&1 | FileCheck %s %}
//
// CHECK-NOT: LEAK

// This test creates a temporary buffer (which is reinterpreted from the main
// application buffers) which is used in kernels but destroyed before
// finalization and execution of the graph. The original buffers lifetime
// extends until after execution of the graph.
// The second run is to check that there are no leaks reported with the embedded
// ZE_DEBUG=4 testing capability.

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
    exp_ext::command_graph Graph{Queue.get_context(), Queue.get_device()};
    buffer<T> BufferA{DataA.data(), range<1>{DataA.size()}};
    buffer<T> BufferB{DataB.data(), range<1>{DataB.size()}};
    buffer<T> BufferC{DataC.data(), range<1>{DataC.size()}};

    Graph.begin_recording(Queue);
    {
      // Create some temporary buffers only for recording
      auto BufferA2 = BufferA.reinterpret<T, 1>(BufferA.get_range());
      auto BufferB2 = BufferB.reinterpret<T, 1>(BufferB.get_range());
      auto BufferC2 = BufferC.reinterpret<T, 1>(BufferC.get_range());

      run_kernels(Queue, Size, BufferA2, BufferB2, BufferC2);
    }
    Graph.end_recording();
    auto GraphExec = Graph.finalize();

    event Event;
    for (unsigned n = 0; n < Iterations; n++) {
      Event = Queue.submit([&](handler &CGH) {
        CGH.depends_on(Event);
        CGH.ext_oneapi_graph(GraphExec);
      });
    }
    // Perform a wait on all graph submissions.
    Queue.wait_and_throw();
  }

  assert(ReferenceA == DataA);
  assert(ReferenceB == DataB);
  assert(ReferenceC == DataC);

  return 0;
}
