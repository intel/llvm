// REQUIRES: level_zero, gpu
// RUN: %{build} -o %t.out
// RUN: %{run} %t.out

// Tests adding buffer 2d copy nodes using the record and replay API
// and submitting the graph.

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
    CGH.parallel_for(range<2>(Size, Size),
                     [=](item<2> id) { AccA[id] += ModValue; });
  });

  // Read & write B
  Queue.submit([&](handler &CGH) {
    auto AccB = BufferB.get_access(CGH);
    CGH.parallel_for(range<2>(Size, Size),
                     [=](item<2> id) { AccB[id] += ModValue; });
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
    CGH.parallel_for(range<2>(Size, Size),
                     [=](item<2> id) { AccB[id] += ModValue; });
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
    for (size_t j = 0; j < Size; j++) {
      assert(ReferenceA[i * Size + j] == HostAccA[i][j]);
      assert(ReferenceB[i * Size + j] == HostAccB[i][j]);
      assert(ReferenceC[i * Size + j] == HostAccC[i][j]);
    }
  }

  return 0;
}
