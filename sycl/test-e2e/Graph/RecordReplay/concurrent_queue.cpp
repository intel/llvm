// REQUIRES: cuda || level_zero, gpu
// RUN: %{build} -o %t.out
// RUN: %{run} %t.out
// Extra run to check for leaks in Level Zero using ZE_DEBUG
// RUN: %if ext_oneapi_level_zero %{env ZE_DEBUG=4 %{run} %t.out 2>&1 | FileCheck %s %}
//
// CHECK-NOT: LEAK

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
