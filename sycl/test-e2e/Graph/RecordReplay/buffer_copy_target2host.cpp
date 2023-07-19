// REQUIRES: level_zero, gpu
// RUN: %{build} -o %t.out
// RUN: %{run} %t.out
// Extra run to check for leaks in Level Zero using ZE_DEBUG
// RUN: %if ext_oneapi_level_zero %{env ZE_DEBUG=4 %{run} %t.out 2>&1 | FileCheck %s %}
//
// CHECK-NOT: LEAK

// Tests adding buffer copy -- Target to Host (read path) --  nodes
// using the record and replay API and submitting the graph.

#include "../graph_common.hpp"

int main() {
  queue Queue;

  using T = int;

  std::vector<T> DataA(Size), DataB(Size);
  std::iota(DataA.begin(), DataA.end(), 1);
  std::iota(DataB.begin(), DataB.end(), 1000);

  std::vector<T> ReferenceA(Size), ReferenceB(Size);
  for (size_t i = 0; i < Size; i++) {
    ReferenceA[i] = DataA[i];
    ReferenceB[i] = DataA[i];
  }

  exp_ext::command_graph Graph{Queue.get_context(), Queue.get_device()};

  buffer<T, 1> BufferA(DataA.data(), range<1>(Size));
  BufferA.set_write_back(false);

  Graph.begin_recording(Queue);

  Queue.submit([&](handler &CGH) {
    auto AccA = BufferA.get_access<access::mode::read>(CGH);
    CGH.copy(AccA, DataB.data());
  });

  Graph.end_recording(Queue);

  auto GraphExec = Graph.finalize();
  Queue.submit([&](handler &CGH) { CGH.ext_oneapi_graph(GraphExec); }).wait();

  for (size_t i = 0; i < Size; i++) {
    assert(ReferenceA[i] == DataA[i]);
    assert(ReferenceB[i] == DataB[i]);
  }
}
