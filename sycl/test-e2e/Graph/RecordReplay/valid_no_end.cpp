// REQUIRES: level_zero, gpu
// RUN: %clangxx -fsycl -fsycl-targets=%sycl_triple %s -o %t.out
// RUN: %GPU_RUN_PLACEHOLDER %t.out

// Tests obtaining a finalized, executable graph from a graph which is
// currently being recorded to without end_recording() being called.

#include "../graph_common.hpp"

int main() {
  queue TestQueue;

  exp_ext::command_graph Graph{TestQueue.get_context(), TestQueue.get_device()};
  {
    queue MyQueue;
    Graph.begin_recording(MyQueue);
  }

  try {
    auto GraphExec = Graph.finalize();
    TestQueue.submit([&](handler &CGH) { CGH.ext_oneapi_graph(GraphExec); });
  } catch (sycl::exception &E) {
    assert(false && "Exception thrown on finalize or submission.\n");
  }
  TestQueue.wait();
  return 0;
}
