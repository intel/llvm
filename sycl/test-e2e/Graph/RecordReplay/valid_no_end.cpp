// RUN: %{build} -o %t.out
// RUN: %{run} %t.out
// Extra run to check for leaks in Level Zero using UR_L0_LEAKS_DEBUG
// RUN: %if level_zero %{env SYCL_PI_LEVEL_ZERO_USE_IMMEDIATE_COMMANDLISTS=0 %{l0_leak_check} %{run} %t.out 2>&1 | FileCheck %s --implicit-check-not=LEAK %}
// Extra run to check for immediate-command-list in Level Zero
// RUN: %if level_zero && linux %{env SYCL_PI_LEVEL_ZERO_USE_IMMEDIATE_COMMANDLISTS=1 %{l0_leak_check} %{run} %t.out 2>&1 | FileCheck %s --implicit-check-not=LEAK %}
//

// Tests obtaining a finalized, executable graph from a graph which is
// currently being recorded to without end_recording() being called.

#include "../graph_common.hpp"

int main() {
  queue Queue;

  exp_ext::command_graph Graph{Queue.get_context(), Queue.get_device()};
  {
    queue MyQueue(Queue.get_context(), Queue.get_device());
    Graph.begin_recording(MyQueue);
  }

  try {
    auto GraphExec = Graph.finalize();
    Queue.submit([&](handler &CGH) { CGH.ext_oneapi_graph(GraphExec); });
  } catch (sycl::exception &E) {
    assert(false && "Exception thrown on finalize or submission.\n");
  }
  Queue.wait();
  return 0;
}
