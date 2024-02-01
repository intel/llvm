// RUN: %{build} -o %t.out
// RUN: %{run} %t.out
// Extra run to check for leaks in Level Zero using UR_L0_LEAKS_DEBUG
// RUN: %if (level_zero && linux) %{env UR_L0_LEAKS_DEBUG=1 %{run} %t.out 2>&1 | FileCheck %s %}
// RUN: %if (level_zero && windows) %{env UR_L0_LEAKS_DEBUG=1 env SYCL_ENABLE_DEFAULT_CONTEXTS=0 %{run} %t.out 2>&1 | FileCheck %s %}
//
// CHECK-NOT: LEAK

// Tests the ability to finalize a command graph while it is currently being
// recorded to.

#include "../graph_common.hpp"

int main() {
  queue Queue{{sycl::ext::intel::property::queue::no_immediate_command_list{}}};

  if (!are_graphs_supported(Queue)) {
    return 0;
  }

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
