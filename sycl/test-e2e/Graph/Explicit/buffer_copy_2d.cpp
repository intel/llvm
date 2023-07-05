// REQUIRES: level_zero, gpu
// RUN: %{build} -o %t.out
// RUN: %{run} %t.out
// RUN: %if ext_oneapi_level_zero %{env ZE_DEBUG=4 %{run} %t.out 2>&1 | FileCheck %s %}
//
// CHECK-NOT: LEAK


// Tests adding buffer 2d copy nodes using the explicit API and submitting
// the graph.
// The second run is to check that there are no leaks reported with the embedded
// ZE_DEBUG=4 testing capability.

#include "../graph_common.hpp"

int main() {
  queue Queue;

  using T = int;

  const T ModValue = 7;
  std::vector<T> DataA(Size * Size), DataB(Size * Size), DataC(Size * Size);

  std::iota(DataA.begin(), DataA.end(), 1);
  std::iota(DataB.begin(), DataB.end(), 10);
  std::iota(DataC.begin(), DataC.end(), 1000);

  // Create reference data for output
  std::vector<T> ReferenceA(DataA), ReferenceB(DataB), ReferenceC(DataC);
  for (unsigned i = 0; i < Iterations; i++) {
    for (size_t j = 0; j < Size * Size; j++) {
      ReferenceA[j] = ReferenceB[j];
      ReferenceA[j] += ModValue;
      ReferenceB[j] = ReferenceA[j];
      ReferenceB[j] += ModValue;
      ReferenceC[j] = ReferenceB[j];
    }
  }

  exp_ext::command_graph Graph{Queue.get_context(), Queue.get_device()};

  // Make the buffers 2D so we can test the rect copy path
  buffer BufferA{DataA.data(), range<2>(Size, Size)};
  BufferA.set_write_back(false);
  buffer BufferB{DataB.data(), range<2>(Size, Size)};
  BufferB.set_write_back(false);
  buffer BufferC{DataC.data(), range<2>(Size, Size)};
  BufferC.set_write_back(false);

  // Copy from B to A
  auto NodeA = Graph.add([&](handler &CGH) {
    auto AccA = BufferA.get_access(CGH);
    auto AccB = BufferB.get_access(CGH);
    CGH.copy(AccB, AccA);
  });

  // Read & write A
  auto NodeB = Graph.add(
      [&](handler &CGH) {
        auto AccA = BufferA.get_access(CGH);
        CGH.parallel_for(range<2>(Size, Size),
                         [=](item<2> id) { AccA[id] += ModValue; });
      },
      {exp_ext::property::node::depends_on(NodeA)});

  // Read & write B
  auto NodeModB = Graph.add(
      [&](handler &CGH) {
        auto AccB = BufferB.get_access(CGH);
        CGH.parallel_for(range<2>(Size, Size),
                         [=](item<2> id) { AccB[id] += ModValue; });
      },
      {exp_ext::property::node::depends_on(NodeA)});

  // memcpy from A to B
  auto NodeC = Graph.add(
      [&](handler &CGH) {
        auto AccA = BufferA.get_access(CGH);
        auto AccB = BufferB.get_access(CGH);
        CGH.copy(AccA, AccB);
      },
      {exp_ext::property::node::depends_on(NodeB, NodeModB)});

  // Read and write B
  auto NodeD = Graph.add(
      [&](handler &CGH) {
        auto AccB = BufferB.get_access(CGH);
        CGH.parallel_for(range<2>(Size, Size),
                         [=](item<2> id) { AccB[id] += ModValue; });
      },
      {exp_ext::property::node::depends_on(NodeC)});

  // Copy from B to C
  Graph.add(
      [&](handler &CGH) {
        auto AccB = BufferB.get_access(CGH);
        auto AccC = BufferC.get_access(CGH);
        CGH.copy(AccB, AccC);
      },
      {exp_ext::property::node::depends_on(NodeD)});

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
    for (size_t j = 0; j < Size; j++) {
      assert(ReferenceA[i * Size + j] == HostAccA[i][j]);
      assert(ReferenceB[i * Size + j] == HostAccB[i][j]);
      assert(ReferenceC[i * Size + j] == HostAccC[i][j]);
    }
  }

  return 0;
}
