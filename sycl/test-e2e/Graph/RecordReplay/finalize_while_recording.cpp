// REQUIRES: level_zero, gpu
// RUN: %clangxx -fsycl -fsycl-targets=%sycl_triple %s -o %t.out
// RUN: %GPU_RUN_PLACEHOLDER %t.out

// Tests the ability to finalize a command graph while it is currently being
// recorded to.

#include "../graph_common.hpp"

int main() {
  queue TestQueue;

  exp_ext::command_graph Graph{TestQueue.get_context(), TestQueue.get_device()};
  Graph.begin_recording(TestQueue);

  try {
    Graph.finalize();
  } catch (sycl::exception &E) {
    assert(false && "Exception thrown on finalize.\n");
  }

  Graph.end_recording();
  return 0;
}
