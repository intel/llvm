// Tests queries associated with graph-owned allocations

#include "../../graph_common.hpp"
#include <sycl/ext/oneapi/experimental/async_alloc/async_alloc.hpp>

using T = int;

int main() {
  queue Queue{};

  size_t Size = 2 << 18;

  // Expected size is number of elements * size of data type * iterations
  const size_t ExpectedMinSize = Size * sizeof(T) * Iterations;

  exp_ext::command_graph Graph{Queue.get_context(), Queue.get_device()};

  // Graph should report 0 memory usage with no allocation nodes
  {
    auto GraphExec = Graph.finalize();
    assert(GraphExec.get_required_mem_size() == 0);
  }

  // Add allocs and frees for each command. Allocs are all root nodes so they
  // will require unique memory for each one.
  for (size_t i = 0; i < Iterations; i++) {
    T *AsyncPtr = nullptr;
    // Add alloc node
    auto AllocNode = add_node(Graph, Queue, [&](handler &CGH) {
      AsyncPtr = static_cast<T *>(
          exp_ext::async_malloc(CGH, usm::alloc::device, Size * sizeof(T)));
    });

    // Free memory, node depends on only the associated allocation node
    add_node(
        Graph, Queue,
        [&](handler &CGH) {
          depends_on_helper(CGH, AllocNode);
          exp_ext::async_free(CGH, AsyncPtr);
        },
        AllocNode);
  }

  auto GraphExec = Graph.finalize();

  // Memory allocated might be adjusted for example based on device granularity,
  // so it may be more than expected but never less.
  assert(GraphExec.get_required_mem_size() >= ExpectedMinSize);

  return 0;
}
