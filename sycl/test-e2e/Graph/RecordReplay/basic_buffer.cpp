// REQUIRES: level_zero, gpu
// RUN: %{build} -o %t.out
// RUN: %{run} %t.out
// RUN: %if ext_oneapi_level_zero %{env ZE_DEBUG=4 %{run} %t.out 2>&1 | FileCheck %s %}
//
// CHECK-NOT: LEAK

// Tests basic queue recording and submission of a graph using buffers for
// inputs and outputs.
// The second run is to check that there are no leaks reported with the embedded
// ZE_DEBUG=4 testing capability.

#include "../graph_common.hpp"

int main() {
  queue Queue;

  using T = unsigned short;

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
    BufferA.set_write_back(false);
    buffer<T> BufferB{DataB.data(), range<1>{DataB.size()}};
    BufferB.set_write_back(false);
    buffer<T> BufferC{DataC.data(), range<1>{DataC.size()}};
    BufferC.set_write_back(false);

    Graph.begin_recording(Queue);
    run_kernels(Queue, Size, BufferA, BufferB, BufferC);
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

    host_accessor HostAccA(BufferA);
    host_accessor HostAccB(BufferB);
    host_accessor HostAccC(BufferC);

    for (size_t i = 0; i < Size; i++) {
      assert(ReferenceA[i] == HostAccA[i]);
      assert(ReferenceB[i] == HostAccB[i]);
      assert(ReferenceC[i] == HostAccC[i]);
    }
  }

  return 0;
}
