// REQUIRES: level_zero, gpu
// RUN: %{build} -o %t.out
// RUN: %{run} %t.out
// Extra run to check for leaks in Level Zero using ZE_DEBUG
// RUN: %if ext_oneapi_level_zero %{env ZE_DEBUG=4 %{run} %t.out 2>&1 | FileCheck %s %}
//
// CHECK-NOT: LEAK

// Tests adding buffer copy offset -- Host to Target (write path) --  nodes
// using the record and replay API and submitting the graph.

#include "../graph_common.hpp"

int main() {
  queue Queue;

  using T = int;

  std::vector<T> DataA(Size + Offset), DataB(Size);
  std::iota(DataA.begin(), DataA.end(), 1);
  std::iota(DataB.begin(), DataB.end(), 1000);

  std::vector<T> ReferenceA(Size + Offset);
  for (size_t i = 0; i < Size + Offset; i++) {
    if (i < Offset)
      ReferenceA[i] = DataA[i];
    else
      ReferenceA[i] = DataB[i - Offset];
  }

  exp_ext::command_graph Graph{Queue.get_context(), Queue.get_device()};

  buffer<T, 1> BufferA(DataA.data(), range<1>(Size + Offset));
  BufferA.set_write_back(false);

  Graph.begin_recording(Queue);

  Queue.submit([&](handler &CGH) {
    auto AccA = BufferA.get_access<access::mode::write>(CGH, range<1>(Size),
                                                        id<1>(Offset));
    CGH.copy(DataB.data(), AccA);
  });

  Graph.end_recording(Queue);

  auto GraphExec = Graph.finalize();
  Queue.submit([&](handler &CGH) { CGH.ext_oneapi_graph(GraphExec); }).wait();

  host_accessor HostAccA(BufferA);

  for (size_t i = 0; i < Size + Offset; i++) {
    assert(ReferenceA[i] == HostAccA[i]);
  }
}
