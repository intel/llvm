// REQUIRES: cuda || level_zero, gpu
// RUN: %{build} -o %t.out
// RUN: %{run} %t.out
// Extra run to check for leaks in Level Zero using ZE_DEBUG
// RUN: %if ext_oneapi_level_zero %{env ZE_DEBUG=4 %{run} %t.out 2>&1 | FileCheck %s %}
//
// CHECK-NOT: LEAK

// Tests the return values from queue graph functions which change the
// internal queue state.

#include "../graph_common.hpp"

int main() {
  queue Queue{{sycl::ext::intel::property::queue::no_immediate_command_list{}}};
  exp_ext::command_graph Graph{Queue.get_context(), Queue.get_device()};

  bool ChangedState = Graph.end_recording();
  assert(ChangedState == false);

  ChangedState = Graph.begin_recording(Queue);
  assert(ChangedState == true);

  ChangedState = Graph.begin_recording(Queue);
  assert(ChangedState == false);

  ChangedState = Graph.end_recording();
  assert(ChangedState == true);

  ChangedState = Graph.end_recording();
  assert(ChangedState == false);

  return 0;
}
