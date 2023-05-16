// REQUIRES: level_zero, gpu
// RUN: %clangxx -fsycl -fsycl-targets=%sycl_triple %s -o %t.out
// RUN: %GPU_RUN_PLACEHOLDER %t.out

// Tests calling finalize() more than once on the same command_graph.

#include "graph_common.hpp"

int main() {
  queue Queue;

  ext::oneapi::experimental::command_graph Graph{Queue.get_context(),
                                                 Queue.get_device()};
  auto GraphExec = Graph.finalize();
  auto GraphExec2 = Graph.finalize();

  return 0;
}
