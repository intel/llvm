// Tests memory reuse behaviour of device graph allocations when an allocation
// is reused multiple times. NOTE: This test partially relies on knowing how the
// implementation works and that the contents of memory will persist when
// allocations are reused. This is useful for testing but is not an assumption
// that a user can or should make.

#include "../../graph_common.hpp"
#include <sycl/ext/oneapi/experimental/async_alloc/async_alloc.hpp>

using T = int;
void add_nodes_to_graph(
    exp_ext::command_graph<exp_ext::graph_state::modifiable> &Graph,
    queue &Queue, size_t Size, T *PtrInput) {
  // Create 1 pointers for async allocations
  T *AsyncPtr = nullptr;
  // Add alloc nodes at the root of the graph for each allocation, this should
  // result in three unique allocations
  auto AllocNode1 = add_node(Graph, Queue, [&](handler &CGH) {
    AsyncPtr = static_cast<T *>(
        exp_ext::async_malloc(CGH, usm::alloc::device, Size * sizeof(T)));
  });

  // Store pointer value for later comparison
  void *FirstAsyncPtr = AsyncPtr;

  // Add kernel that fills the async alloc with values
  auto KernelFillPtrs = add_node(
      Graph, Queue,
      [&](handler &CGH) {
        depends_on_helper(CGH, {AllocNode1});
        CGH.parallel_for(range<1>(Size), [=](item<1> Item) {
          size_t LinID = Item.get_linear_id();
          AsyncPtr[LinID] = (1 + LinID);
        });
      },
      AllocNode1);
  // Free all the async allocation, making it possible to reuse it
  auto FreeNode1 = add_node(
      Graph, Queue,
      [&](handler &CGH) {
        depends_on_helper(CGH, KernelFillPtrs);
        exp_ext::async_free(CGH, AsyncPtr);
      },
      KernelFillPtrs);

  // Add an allocation node which should reuse the previous allocation, followed
  // by a kernel that uses the data, and then freeing the allocation.

  auto AllocNode2 = add_node(
      Graph, Queue,
      [&](handler &CGH) {
        depends_on_helper(CGH, {FreeNode1});
        AsyncPtr = static_cast<T *>(
            exp_ext::async_malloc(CGH, usm::alloc::device, Size * sizeof(T)));
      },
      FreeNode1);

  // Check that the new allocation matches one of the previously returned
  // values.
  assert(AsyncPtr == FirstAsyncPtr);

  auto KernelAdd1 = add_node(
      Graph, Queue,
      [&](handler &CGH) {
        depends_on_helper(CGH, {AllocNode2});
        CGH.parallel_for(range<1>(Size), [=](item<1> Item) {
          size_t LinID = Item.get_linear_id();
          PtrInput[LinID] += AsyncPtr[LinID];
        });
      },
      AllocNode2);

  auto FreeNode2 = add_node(
      Graph, Queue,
      [&](handler &CGH) {
        depends_on_helper(CGH, KernelAdd1);
        exp_ext::async_free(CGH, AsyncPtr);
      },
      KernelAdd1);

  // Repeat the previous 3 nodes to test reuse an allocation which has
  // previously been reused.

  auto AllocNode3 = add_node(
      Graph, Queue,
      [&](handler &CGH) {
        depends_on_helper(CGH, {FreeNode2});
        AsyncPtr = static_cast<T *>(
            exp_ext::async_malloc(CGH, usm::alloc::device, Size * sizeof(T)));
      },
      FreeNode2);

  // Check that the new allocation matches one of the previously returned
  // values.
  assert(AsyncPtr == FirstAsyncPtr);

  auto KernelAdd2 = add_node(
      Graph, Queue,
      [&](handler &CGH) {
        depends_on_helper(CGH, {AllocNode3});
        CGH.parallel_for(range<1>(Size), [=](item<1> Item) {
          size_t LinID = Item.get_linear_id();
          PtrInput[LinID] += AsyncPtr[LinID];
        });
      },
      AllocNode3);

  add_node(
      Graph, Queue,
      [&](handler &CGH) {
        depends_on_helper(CGH, KernelAdd2);
        exp_ext::async_free(CGH, AsyncPtr);
      },
      KernelAdd2);
}

void calculate_reference_data(size_t Iterations, size_t Size,
                              std::vector<T> &ReferenceOutput) {
  for (size_t i = 0; i < Iterations; i++) {
    for (size_t j = 0; j < Size; j++) {
      ReferenceOutput[j] += (j + 1) * 2;
    }
  }
}

int main() {
  queue Queue{};

  std::vector<T> DataInput(Size);

  std::iota(DataInput.begin(), DataInput.end(), 1);

  std::vector<T> ReferenceOutput(DataInput);
  calculate_reference_data(Iterations, Size, ReferenceOutput);

  exp_ext::command_graph Graph{Queue.get_context(), Queue.get_device()};

  T *PtrInput = malloc_device<T>(Size, Queue);

  std::vector<T> OutputData(Size);

  Queue.copy(DataInput.data(), PtrInput, Size);
  Queue.wait_and_throw();

  // Add commands to graph
  add_nodes_to_graph(Graph, Queue, Size, PtrInput);

  auto GraphExec = Graph.finalize();

  for (unsigned n = 0; n < Iterations; n++) {
    Queue.submit([&](handler &CGH) { CGH.ext_oneapi_graph(GraphExec); });
  }

  Queue.wait_and_throw();
  Queue.copy(PtrInput, OutputData.data(), Size).wait_and_throw();
  for (size_t i = 0; i < Size; i++) {
    assert(check_value(i, ReferenceOutput[i], OutputData[i], "OutputData"));
  }

  free(PtrInput, Queue);

  return 0;
}
