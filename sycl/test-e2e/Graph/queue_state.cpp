// REQUIRES: level_zero, gpu
// RUN: %clangxx -fsycl -fsycl-targets=%sycl_triple %s -o %t.out
// RUN: %GPU_RUN_PLACEHOLDER %t.out

// Expected fail as this query isn't implemented yet
// XFAIL: *

// Tests the return values from queue graph functions which change the
// internal queue state.

#include "graph_common.hpp"

int main() {
  queue TestQueue;

  exp_ext::queue_state State = TestQueue.get_info<info::queue::state>();
  assert(State == exp_ext::queue_state::executing);

  exp_ext::command_graph Graph{TestQueue.get_context(), TestQueue.get_device()};
  Graph.begin_recording(TestQueue);
  State = TestQueue.get_info<info::queue::state>();
  assert(State == exp_ext::queue_state::recording);

  Graph.end_recording();
  State = TestQueue.get_info<info::queue::state>();
  assert(State == exp_ext::queue_state::executing);

  return 0;
}
