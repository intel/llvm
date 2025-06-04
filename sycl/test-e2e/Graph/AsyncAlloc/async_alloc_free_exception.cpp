// RUN: %{build} -o %t.out
// RUN: %{run} %t.out

// Tests that attempting to free a pointer in a graph that doesn't have an
// associated allocation node fails in a range of scenarios

#include "../graph_common.hpp"
#include <sycl/ext/oneapi/experimental/async_alloc/async_alloc.hpp>

using T = int;

// Attempts to add a free to the graph for a pointer with no associated
// allocation, and check that the correct exception is returned.
void addInvalidFreeAndCheckForException(
    exp_ext::command_graph<exp_ext::graph_state::modifiable> &Graph) {
  void *FakePtr = (void *)1;
  std::error_code ExceptionCode = make_error_code(sycl::errc::success);
  try {
    Graph.add([&](handler &CGH) { exp_ext::async_free(CGH, FakePtr); });
  } catch (const exception &e) {
    ExceptionCode = e.code();
  }

  assert(ExceptionCode == sycl::errc::invalid);
}

int main() {
  queue Queue{};

  exp_ext::command_graph Graph{Queue.get_context(), Queue.get_device()};

  // Add an invalid free to the graph
  addInvalidFreeAndCheckForException(Graph);

  void *AsyncPtr = nullptr;
  // Add a real allocation node
  auto AllocNode = Graph.add([&](handler &CGH) {
    AsyncPtr = sycl::ext::oneapi::experimental::async_malloc(
        CGH, usm::alloc::device, Size);
  });

  // Try the invalid free again
  addInvalidFreeAndCheckForException(Graph);

  // Add a real free node
  Graph.add([&](handler &CGH) { exp_ext::async_free(CGH, AsyncPtr); });

  // Try the invalid free again
  addInvalidFreeAndCheckForException(Graph);

  return 0;
}
