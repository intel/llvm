// REQUIRES: level_zero, gpu
// RUN: %{build} -o %t.out
// RUN: %{run} %t.out
// Extra run to check for leaks in Level Zero using ZE_DEBUG
// RUN: %if ext_oneapi_level_zero %{env ZE_DEBUG=4 %{run} %t.out 2>&1 | FileCheck %s %}
//
// CHECK-NOT: LEAK

// Tests adding buffer copy nodes with offsets
// using the record and replay API and submitting the graph.
// The second run is to check that there are no leaks reported with the embedded
// ZE_DEBUG=4 testing capability.

#include "../graph_common.hpp"

int main() {
  queue Queue;

  using T = int;

  size_t OffsetSrc = 2 * size_t(Size / 4);
  size_t OffsetDst = size_t(Size / 4);
  std::vector<T> DataA(Size), DataB(Size);

  std::iota(DataA.begin(), DataA.end(), 1);
  std::iota(DataB.begin(), DataB.end(), 10000);

  // Create reference data for output
  std::vector<T> ReferenceA(DataA), ReferenceB(DataB);
  for (size_t j = 0; j < Size; j++) {
    ReferenceA[j] = DataA[j];
    ReferenceB[j] = DataB[j];
  }
  for (size_t j = OffsetDst; j < Size - (OffsetSrc - OffsetDst); j++) {
    ReferenceB[j] = DataA[(j - OffsetDst) + OffsetSrc];
  }

  exp_ext::command_graph Graph{Queue.get_context(), Queue.get_device()};

  buffer BufferA{DataA};
  BufferA.set_write_back(false);
  buffer BufferB{DataB};
  BufferB.set_write_back(false);

  Graph.begin_recording(Queue);

  // Copy from A to B
  Queue.submit([&](handler &CGH) {
    auto AccA = BufferA.get_access<access::mode::read_write>(
        CGH, range<1>(Size - OffsetSrc), id<1>(OffsetSrc));
    auto AccB = BufferB.get_access<access::mode::read_write>(
        CGH, range<1>(Size - OffsetDst), id<1>(OffsetDst));
    CGH.copy(AccA, AccB);
  });

  Graph.end_recording(Queue);

  auto GraphExec = Graph.finalize();
  Queue.submit([&](handler &CGH) { CGH.ext_oneapi_graph(GraphExec); }).wait();

  host_accessor HostAccA(BufferA);
  host_accessor HostAccB(BufferB);

  for (size_t i = 0; i < Size; i++) {
    assert(ReferenceA[i] == HostAccA[i]);
    assert(ReferenceB[i] == HostAccB[i]);
  }

  return 0;
}
