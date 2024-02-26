// RUN: %{build} -o %t.out
// RUN: %{run} %t.out
// Extra run to check for leaks in Level Zero using UR_L0_LEAKS_DEBUG
// RUN: %if level_zero %{env SYCL_PI_LEVEL_ZERO_USE_IMMEDIATE_COMMANDLISTS=0 %{l0_leak_check} %{run} %t.out 2>&1 | FileCheck %s --implicit-check-not=LEAK %}
// Extra run to check for immediate-command-list in Level Zero
// RUN: %if level_zero && linux %{env SYCL_PI_LEVEL_ZERO_USE_IMMEDIATE_COMMANDLISTS=1 %{l0_leak_check} %{run} %t.out 2>&1 | FileCheck %s --implicit-check-not=LEAK %}
//

// Tests attempting to add a node to a command_graph while it is being
// recorded to by a queue is an error.

#include "../graph_common.hpp"

int main() {
  queue Queue{};

  if (!are_graphs_supported(Queue)) {
    return 0;
  }

  bool Success = false;

  exp_ext::command_graph Graph{Queue.get_context(), Queue.get_device()};
  Graph.begin_recording(Queue);

  try {
    Graph.add([&](handler &CGH) {});
  } catch (sycl::exception &E) {
    auto StdErrc = E.code().value();
    if (StdErrc == static_cast<int>(errc::invalid)) {
      Success = true;
    }
  }
  assert(Success);

  Success = false;
  try {
    Graph.add({});
  } catch (sycl::exception &E) {
    auto StdErrc = E.code().value();
    Success = (StdErrc == static_cast<int>(errc::invalid));
  }
  assert(Success);

  Graph.end_recording();
  return 0;
}
