// REQUIRES: level_zero, gpu
// RUN: %{build} -o %t.out
// RUN: %{run} %t.out
// RUN: %if ext_oneapi_level_zero %{env ZE_DEBUG=4 %{run} %t.out 2>&1 | FileCheck %s %}
//
// CHECK-NOT: LEAK

// Expected fail as exception not yet implemented
// XFAIL: *

// Tests that attempting to record to a command_graph when it is already being
// being recorded to by another queue throws an exception.
// The second run is to check that there are no leaks reported with the embedded
// ZE_DEBUG=4 testing capability.

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
