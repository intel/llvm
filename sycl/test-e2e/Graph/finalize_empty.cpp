// REQUIRES: level_zero, gpu
// RUN: %clangxx -fsycl -fsycl-targets=%sycl_triple %s -o %t.out
// RUN: %GPU_RUN_PLACEHOLDER %t.out

// Tests the ability to finalize a command graph without recording any nodes.

#include "graph_common.hpp"

int main() {
  queue TestQueue;

  ext::oneapi::experimental::command_graph Graph{TestQueue.get_context(),
                                                 TestQueue.get_device()};
  auto GraphExec = Graph.finalize();

  TestQueue.submit([&](handler &CGH) { CGH.ext_oneapi_graph(GraphExec); });
  TestQueue.wait_and_throw();

  return 0;
}
