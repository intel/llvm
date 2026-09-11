// SYCL_LAUNCH_BLOCKING must not change what a graph computes: adding a node
// does not execute it, so it must not drain the queue, while submitting a
// finalized graph is a regular, blocking submission. Every node accumulates
// into the same allocation, so a dropped or repeated node is detected.

#include "../graph_common.hpp"

int main() {
  queue Queue{};

  exp_ext::command_graph Graph{Queue.get_context(), Queue.get_device()};

  int *Ptr = malloc_shared<int>(Size, Queue);
  Queue.fill(Ptr, int{0}, Size).wait_and_throw();

  unsigned HostTaskRuns = 0;

  auto NodeA = add_node(Graph, Queue, [&](handler &CGH) {
    CGH.parallel_for(range<1>(Size), [=](item<1> id) { Ptr[id] += 1; });
  });

  // A graph containing a host task takes a different submission path, since
  // host task dependencies cannot be expressed natively.
  auto NodeB = add_node(
      Graph, Queue,
      [&](handler &CGH) {
        depends_on_helper(CGH, NodeA);
        CGH.host_task([&]() { ++HostTaskRuns; });
      },
      NodeA);

  add_node(
      Graph, Queue,
      [&](handler &CGH) {
        depends_on_helper(CGH, NodeB);
        CGH.parallel_for(range<1>(Size), [=](item<1> id) { Ptr[id] += 1; });
      },
      NodeB);

  auto GraphExec = Graph.finalize();

  // Replaying the same executable graph repeatedly must keep working, through
  // both the queue shortcut and a command group.
  for (unsigned n = 0; n < Iterations; n++) {
    Queue.ext_oneapi_graph(GraphExec);
    Queue.submit([&](handler &CGH) { CGH.ext_oneapi_graph(GraphExec); });
  }
  Queue.wait_and_throw();

  const int Reference = 2 * 2 * Iterations;
  for (size_t i = 0; i < Size; i++) {
    assert(check_value(i, Reference, Ptr[i], "Ptr"));
  }
  assert(HostTaskRuns == 2 * Iterations);

  free(Ptr, Queue);
  return 0;
}
