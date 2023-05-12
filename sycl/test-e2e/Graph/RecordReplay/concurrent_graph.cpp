// REQUIRES: level_zero, gpu
// RUN: %clangxx -fsycl -fsycl-targets=%sycl_triple %s -o %t.out
// RUN: %GPU_RUN_PLACEHOLDER %t.out

// Expected fail as exception not yet implemented
// XFAIL: *

// Tests that attempting to record to a command_graph when it is already being
// being recorded to by another queue throws an exception.

#include "../graph_common.hpp"

int main() {
  queue TestQueue;

  bool Success = false;

  exp_ext::command_graph Graph{TestQueue.get_context(), TestQueue.get_device()};
  Graph.begin_recording(TestQueue);

  queue TestQueue2;
  try {
    Graph.begin_recording(TestQueue2);
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
