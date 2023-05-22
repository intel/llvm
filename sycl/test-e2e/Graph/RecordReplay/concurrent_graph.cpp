// REQUIRES: level_zero, gpu
// RUN: %{build} -o %t.out
// RUN: %{run} %t.out

// Expected fail as exception not yet implemented
// XFAIL: *

// Tests that attempting to record to a command_graph when it is already being
// being recorded to by another queue throws an exception.

#include "../graph_common.hpp"

int main() {
  queue Queue;

  bool Success = false;

  exp_ext::command_graph Graph{Queue.get_context(), Queue.get_device()};
  Graph.begin_recording(Queue);

  queue Queue2;
  try {
    Graph.begin_recording(Queue2);
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
