// REQUIRES: level_zero, gpu
// RUN: %{build} -o %t.out
// RUN: %{run} %t.out
// RUN: %if ext_oneapi_level_zero %{env ZE_DEBUG=4 %{run} %t.out 2>&1 | FileCheck %s %}
//
// CHECK-NOT: LEAK

// Tests adding buffer copy nodes using the record and replay API and submitting
// the graph.
// The second run is to check that there are no leaks reported with the embedded
// ZE_DEBUG=4 testing capability.

#include "../graph_common.hpp"

int main() {
  queue Queue;

  using T = int;

  const T ModValue = 7;
  std::vector<T> DataA(Size), DataB(Size), DataC(Size);

  std::iota(DataA.begin(), DataA.end(), 1);
  std::iota(DataB.begin(), DataB.end(), 10);
  std::iota(DataC.begin(), DataC.end(), 1000);

  // Create reference data for output
  std::vector<T> ReferenceA(DataA), ReferenceB(DataB), ReferenceC(DataC);
  for (unsigned i = 0; i < Iterations; i++) {
    for (size_t j = 0; j < Size; j++) {
      ReferenceA[j] = ReferenceB[j];
      ReferenceA[j] += ModValue;
      ReferenceB[j] = ReferenceA[j];
      ReferenceB[j] += ModValue;
      ReferenceC[j] = ReferenceB[j];
    }
  }

  exp_ext::command_graph Graph{Queue.get_context(), Queue.get_device()};

  buffer BufferA{DataA};
  BufferA.set_write_back(false);
  buffer BufferB{DataB};
  BufferB.set_write_back(false);
  buffer BufferC{DataC};
  BufferC.set_write_back(false);

  Graph.begin_recording(Queue);

  // Copy from B to A
  Queue.submit([&](handler &CGH) {
    auto AccA = BufferA.get_access(CGH);
    auto AccB = BufferB.get_access(CGH);
    CGH.copy(AccB, AccA);
  });

  // Read & write A
  Queue.submit([&](handler &CGH) {
    auto AccA = BufferA.get_access(CGH);
    CGH.parallel_for(range<1>(Size), [=](item<1> id) {
      auto LinID = id.get_linear_id();
      AccA[LinID] += ModValue;
    });
  });

  // Read & write B
  Queue.submit([&](handler &CGH) {
    auto AccB = BufferB.get_access(CGH);
    CGH.parallel_for(range<1>(Size), [=](item<1> id) {
      auto LinID = id.get_linear_id();
      AccB[LinID] += ModValue;
    });
  });

  // memcpy from A to B
  Queue.submit([&](handler &CGH) {
    auto AccA = BufferA.get_access(CGH);
    auto AccB = BufferB.get_access(CGH);
    CGH.copy(AccA, AccB);
  });

  // Read and write B
  Queue.submit([&](handler &CGH) {
    auto AccB = BufferB.get_access(CGH);
    CGH.parallel_for(range<1>(Size), [=](item<1> id) {
      auto LinID = id.get_linear_id();
      AccB[LinID] += ModValue;
    });
  });

  // Copy from B to C
  Queue.submit([&](handler &CGH) {
    auto AccB = BufferB.get_access(CGH);
    auto AccC = BufferC.get_access(CGH);
    CGH.copy(AccB, AccC);
  });

  Graph.end_recording(Queue);

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

  return 0;
}
