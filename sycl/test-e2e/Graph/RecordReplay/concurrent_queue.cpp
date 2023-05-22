// REQUIRES: level_zero, gpu
// RUN: %{build} -o %t.out
// RUN: %{run} %t.out

// Tests attempting to begin recording to a graph when recording is
// already in progress on another graph throws an error.

#include "../graph_common.hpp"

int main() {
  queue Queue;

  bool Success = false;

  exp_ext::command_graph GraphA{Queue.get_context(), Queue.get_device()};
  GraphA.begin_recording(Queue);

  try {
    exp_ext::command_graph GraphB{Queue.get_context(), Queue.get_device()};
    GraphB.begin_recording(Queue);
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
