// REQUIRES: level_zero, gpu
// RUN: %{build} -o %t.out
// RUN: %{run} %t.out

// Tests adding a buffer copy node using the explicit API and submitting
// the graph.

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
        CGH.parallel_for(range<1>(Size), [=](item<1> id) {
          auto LinID = id.get_linear_id();
          AccA[LinID] += ModValue;
        });
      },
      {exp_ext::property::node::depends_on(NodeA)});

  // Read & write B
  auto NodeModB = Graph.add(
      [&](handler &CGH) {
        auto AccB = BufferB.get_access(CGH);
        CGH.parallel_for(range<1>(Size), [=](item<1> id) {
          auto LinID = id.get_linear_id();
          AccB[LinID] += ModValue;
        });
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
        CGH.parallel_for(range<1>(Size), [=](item<1> id) {
          auto LinID = id.get_linear_id();
          AccB[LinID] += ModValue;
        });
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
    assert(ReferenceA[i] == HostAccA[i]);
    assert(ReferenceB[i] == HostAccB[i]);
    assert(ReferenceC[i] == HostAccC[i]);
  }

  return 0;
}
