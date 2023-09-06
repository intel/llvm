// REQUIRES: cuda || level_zero, gpu
// RUN: %{build} -o %t.out
// RUN: %{run} %t.out
// Extra run to check for leaks in Level Zero using ZE_DEBUG
// RUN: %if ext_oneapi_level_zero %{env ZE_DEBUG=4 %{run} %t.out 2>&1 | FileCheck %s %}
//
// CHECK-NOT: LEAK

// Tests attempting to add a node to a command_graph while it is being
// recorded to by a queue is an error.

#include "../graph_common.hpp"

int main() {
  queue Queue{{sycl::ext::intel::property::queue::no_immediate_command_list{}}};

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
