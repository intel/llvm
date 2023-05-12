// REQUIRES: level_zero, gpu
// RUN: %clangxx -fsycl -fsycl-targets=%sycl_triple %s -o %t.out
// RUN: %GPU_RUN_PLACEHOLDER %t.out

// Tests attempting to begin recording to a graph when recording is
// already in progress on another graph throws an error.

#include "../graph_common.hpp"

int main() {
  queue TestQueue;

  bool Success = false;

  exp_ext::command_graph GraphA{TestQueue.get_context(),
                                TestQueue.get_device()};
  GraphA.begin_recording(TestQueue);

  try {
    exp_ext::command_graph GraphB{TestQueue.get_context(),
                                  TestQueue.get_device()};
    GraphB.begin_recording(TestQueue);
  } catch (sycl::exception &E) {
    auto StdErrc = E.code().value();
    if (StdErrc == static_cast<int>(errc::invalid)) {
      Success = true;
    }
  }

  GraphA.end_recording();

  assert(Success);
  return 0;
}
