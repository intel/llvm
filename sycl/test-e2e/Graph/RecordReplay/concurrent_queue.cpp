// RUN: %{build} -o %t.out
// RUN: %{run} %t.out
// Extra run to check for leaks in Level Zero using UR_L0_LEAKS_DEBUG
// RUN: %if level_zero %{env SYCL_PI_LEVEL_ZERO_USE_IMMEDIATE_COMMANDLISTS=0 %{l0_leak_check} %{run} %t.out 2>&1 | FileCheck %s --implicit-check-not=LEAK %}
// Extra run to check for immediate-command-list in Level Zero
// RUN: %if level_zero && linux %{env SYCL_PI_LEVEL_ZERO_USE_IMMEDIATE_COMMANDLISTS=1 %{l0_leak_check} %{run} %t.out 2>&1 | FileCheck %s --implicit-check-not=LEAK %}
//

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
    Success = E.code() == errc::invalid;
  }

  GraphA.end_recording();

  assert(Success);
  return 0;
}
