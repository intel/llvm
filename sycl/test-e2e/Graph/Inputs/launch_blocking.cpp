// SYCL_LAUNCH_BLOCKING must not change what a graph computes: adding a node
// does not execute it, so it must not drain the queue, while submitting a
// finalized graph is a regular, blocking submission.

#include "../graph_common.hpp"

// A host task node splits the graph into partitions that are all
// enqueued in one call, while the executable graph's write lock is held.
// Host tasks require internal synchronization to enqueue the next partition,
// so the host task case tests that SYCL_LAUNCH_BLOCKING does not deadlock
// with the internal synchronization.
static void runGraph(queue &Queue, bool WithHostTask) {
  exp_ext::command_graph Graph{Queue.get_context(), Queue.get_device()};

  int *Ptr = malloc_shared<int>(Size, Queue);
  Queue.fill(Ptr, int{0}, Size);

  unsigned HostTaskRuns = 0;

  auto NodeA = add_node(Graph, Queue, [&](handler &CGH) {
    CGH.parallel_for(range<1>(Size), [=](item<1> id) { Ptr[id] += 1; });
  });

  auto Dep = NodeA;
  if (WithHostTask)
    Dep = add_node(
        Graph, Queue,
        [&](handler &CGH) {
          depends_on_helper(CGH, NodeA);
          CGH.host_task([&]() { ++HostTaskRuns; });
        },
        NodeA);

  add_node(
      Graph, Queue,
      [&](handler &CGH) {
        depends_on_helper(CGH, Dep);
        CGH.parallel_for(range<1>(Size), [=](item<1> id) { Ptr[id] += 1; });
      },
      Dep);

  auto GraphExec = Graph.finalize();

  // Replaying the same executable graph repeatedly must keep working, through
  // both the queue shortcut and a command group.
  for (unsigned n = 0; n < Iterations; n++) {
    Queue.ext_oneapi_graph(GraphExec);
    Queue.submit([&](handler &CGH) { CGH.ext_oneapi_graph(GraphExec); });
  }

  // Host tasks are not made synchronous, so we have to explicitly wait.
  if (WithHostTask)
    Queue.wait_and_throw();

  const int Reference = 2 * 2 * Iterations;
  for (size_t i = 0; i < Size; i++) {
    assert(check_value(i, Reference, Ptr[i], "Ptr"));
  }
  assert(HostTaskRuns == (WithHostTask ? 2 * Iterations : 0));

  free(Ptr, Queue);
}

int main() {
  queue Queue;
  runGraph(Queue, /*WithHostTask=*/false);
  runGraph(Queue, /*WithHostTask=*/true);
  return 0;
}
