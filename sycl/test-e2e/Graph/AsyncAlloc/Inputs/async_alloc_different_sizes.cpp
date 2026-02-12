// Tests async allocations with different sizes.

#include "../../graph_common.hpp"
#include <sycl/ext/oneapi/experimental/async_alloc/async_alloc.hpp>

void asyncAllocWorksWithSize(size_t Size) {
  queue Queue{};

  exp_ext::command_graph Graph{Queue.get_context(), Queue.get_device()};

  void *AsyncPtr = nullptr;
  // Add alloc node
  auto AllocNode = add_node(Graph, Queue, [&](handler &CGH) {
    AsyncPtr = exp_ext::async_malloc(CGH, usm::alloc::device, Size);
  });

  add_node(
      Graph, Queue,
      [&](handler &CGH) {
        depends_on_helper(CGH, AllocNode);
        exp_ext::async_free(CGH, AsyncPtr);
      },
      AllocNode);

  auto GraphExec = Graph.finalize();
}

int main() {
  asyncAllocWorksWithSize(1);
  asyncAllocWorksWithSize(131);
  asyncAllocWorksWithSize(10071);
  asyncAllocWorksWithSize(1007177);
  asyncAllocWorksWithSize(191439360);
}
