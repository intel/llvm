// Tests zero initializing a graph memory allocation based on memory pool
// properties

#include "../../graph_common.hpp"
#include <sycl/ext/oneapi/experimental/async_alloc/async_alloc.hpp>
#include <sycl/ext/oneapi/experimental/async_alloc/memory_pool.hpp>
#include <sycl/ext/oneapi/experimental/async_alloc/memory_pool_properties.hpp>

using T = int;

int main() {
  queue Queue{};

  std::vector<T> Output(Size);
  std::vector<T> ReferenceOutput(Size, T(0));

  exp_ext::command_graph Graph{Queue.get_context(), Queue.get_device()};

  exp_ext::memory_pool MemPool{
      Queue, usm::alloc::device, {exp_ext::property::memory_pool::zero_init{}}};

  // Add commands to graph
  T *AsyncPtrA = nullptr;
  // Add alloc node that is zero initialized
  auto AllocNode = add_node(Graph, Queue, [&](handler &CGH) {
    AsyncPtrA = static_cast<T *>(
        exp_ext::async_malloc_from_pool(CGH, Size * sizeof(T), MemPool));
  });

  // Copy that zero init memory back to host
  auto MemcpyNode = add_node(
      Graph, Queue,
      [&](handler &CGH) {
        depends_on_helper(CGH, AllocNode);
        CGH.memcpy(Output.data(), AsyncPtrA, Size * sizeof(T));
      },
      AllocNode);

  // Free memory
  add_node(
      Graph, Queue,
      [&](handler &CGH) {
        depends_on_helper(CGH, MemcpyNode);
        exp_ext::async_free(CGH, AsyncPtrA);
      },
      MemcpyNode);

  auto GraphExec = Graph.finalize();

  Queue.submit([&](handler &CGH) { CGH.ext_oneapi_graph(GraphExec); });

  Queue.wait_and_throw();

  for (size_t i = 0; i < Size; i++) {
    assert(check_value(i, ReferenceOutput[i], Output[i], "Output"));
  }

  return 0;
}
