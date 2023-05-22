// REQUIRES: level_zero, gpu
// RUN: %{build} -o %t.out
// RUN: %{run} %t.out

// Tests the ability to finalize a command graph while it is currently being
// recorded to.

#include "../graph_common.hpp"

int main() {
  queue Queue;

  exp_ext::command_graph Graph{Queue.get_context(), Queue.get_device()};
  Graph.begin_recording(Queue);

  try {
    Graph.finalize();
  } catch (sycl::exception &E) {
    assert(false && "Exception thrown on finalize.\n");
  }

  Graph.end_recording();
  return 0;
}
