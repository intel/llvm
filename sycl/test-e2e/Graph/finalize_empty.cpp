// REQUIRES: level_zero, gpu
// RUN: %{build} -o %t.out
// RUN: %{run} %t.out

// Tests the ability to finalize a command graph without recording any nodes.

#include "graph_common.hpp"

int main() {
  queue Queue;

  ext::oneapi::experimental::command_graph Graph{Queue.get_context(),
                                                 Queue.get_device()};
  auto GraphExec = Graph.finalize();

  Queue.submit([&](handler &CGH) { CGH.ext_oneapi_graph(GraphExec); });
  Queue.wait_and_throw();

  return 0;
}
