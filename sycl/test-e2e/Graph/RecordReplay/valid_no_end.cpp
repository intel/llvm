// REQUIRES: level_zero, gpu
// RUN: %{build} -o %t.out
// RUN: %{run} %t.out

// Tests obtaining a finalized, executable graph from a graph which is
// currently being recorded to without end_recording() being called.

#include "../graph_common.hpp"

int main() {
  queue Queue;

  exp_ext::command_graph Graph{Queue.get_context(), Queue.get_device()};
  {
    queue MyQueue;
    Graph.begin_recording(MyQueue);
  }

  try {
    auto GraphExec = Graph.finalize();
    Queue.submit([&](handler &CGH) { CGH.ext_oneapi_graph(GraphExec); });
  } catch (sycl::exception &E) {
    assert(false && "Exception thrown on finalize or submission.\n");
  }
  Queue.wait();
  return 0;
}
