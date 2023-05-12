// REQUIRES: level_zero, gpu
// RUN: %clangxx -fsycl -fsycl-targets=%sycl_triple %s -o %t.out
// RUN: %GPU_RUN_PLACEHOLDER %t.out

// Expected Fail as exception not implemented yet
// XFAIL: *

// Tests attempting to add a node to a command_graph while it is being
// recorded to by a queue is an error.

#include "../graph_common.hpp"

int main() {
  queue TestQueue;

  bool Success = false;

  exp_ext::command_graph Graph{TestQueue.get_context(), TestQueue.get_device()};
  Graph.begin_recording(TestQueue);

  try {
    Graph.add([&](handler &CGH) {});
  } catch (sycl::exception &E) {
    auto StdErrc = E.code().value();
    if (StdErrc == static_cast<int>(errc::invalid)) {
      Success = true;
    }
  }

  Graph.end_recording();
  assert(Success);
  return 0;
}
