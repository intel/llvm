// REQUIRES: level_zero, gpu
// RUN: %{build} -o %t.out
// RUN: %{run} %t.out
// RUN: %if ext_oneapi_level_zero %{env ZE_DEBUG=4 %{run} %t.out 2>&1 | FileCheck %s %}
//
// CHECK-NOT: LEAK

// Tests adding buffer copy -- Host to Target (write path) --  nodes
// using the explicit API and submitting the graph.
// The second run is to check that there are no leaks reported with the embedded
// ZE_DEBUG=4 testing capability.

#include "../graph_common.hpp"

int main() {
  queue Queue;

  using T = int;

  std::vector<T> DataA(Size), DataB(Size);
  std::iota(DataA.begin(), DataA.end(), 1);
  std::iota(DataB.begin(), DataB.end(), 1000);

  std::vector<T> ReferenceA(Size);
  for (size_t i = 0; i < Size; i++) {
    ReferenceA[i] = DataB[i];
  }

  exp_ext::command_graph Graph{Queue.get_context(), Queue.get_device()};

  buffer<T, 1> BufferA(DataA.data(), range<1>(Size));
  BufferA.set_write_back(false);

  auto NodeA = Graph.add([&](handler &CGH) {
    auto AccA = BufferA.get_access(CGH);
    CGH.copy(DataB.data(), AccA);
  });

  auto GraphExec = Graph.finalize();
  Queue.submit([&](handler &CGH) { CGH.ext_oneapi_graph(GraphExec); }).wait();

  host_accessor HostAccA(BufferA);

  for (size_t i = 0; i < Size; i++) {
    assert(ReferenceA[i] == HostAccA[i]);
  }
}
