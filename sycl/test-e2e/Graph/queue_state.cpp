// REQUIRES: level_zero, gpu
// RUN: %{build} -o %t.out
// RUN: %{run} %t.out

// Tests the return values from queue graph functions which change the
// internal queue state.

#include "graph_common.hpp"

int main() {
  queue Queue;

  exp_ext::queue_state State = Queue.ext_oneapi_get_state();
  assert(State == exp_ext::queue_state::executing);

  exp_ext::command_graph Graph{Queue.get_context(), Queue.get_device()};
  Graph.begin_recording(Queue);
  State = Queue.ext_oneapi_get_state();
  assert(State == exp_ext::queue_state::recording);

  Graph.end_recording();
  State = Queue.ext_oneapi_get_state();
  assert(State == exp_ext::queue_state::executing);

  return 0;
}
