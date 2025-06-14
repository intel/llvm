// Tests memory reuse behaviour of device graph allocations from a zero-init
// memory pool.
// NOTE: This test partially relies on knowing how the implementation works and
// that the contents of memory will persist when allocations are reused. This is
// useful for testing but is not an assumption that a user can or should make.

#include "../../graph_common.hpp"
#include <sycl/ext/oneapi/experimental/async_alloc/async_alloc.hpp>
#include <sycl/ext/oneapi/experimental/async_alloc/memory_pool.hpp>
#include <sycl/ext/oneapi/experimental/async_alloc/memory_pool_properties.hpp>

#define __SYCL_USE_VARIADIC_SPIRV_OCL_PRINTF__

using T = int;
void add_nodes_to_graph(
    exp_ext::command_graph<exp_ext::graph_state::modifiable> &Graph,
    queue &Queue, size_t Size, T *PtrInput) {
  // Create a memory pool for async allocations with the zero init property
  exp_ext::memory_pool MemPool{Queue, usm::alloc::device,
                               exp_ext::properties{exp_ext::zero_init()}};
  // Create 2 pointers for async allocations
  T *AsyncPtrA = nullptr;
  T *AsyncPtrB = nullptr;
  // Add alloc nodes at the root of the graph for each allocation, this should
  // result in three unique allocations
  auto AllocNodeA = add_node(Graph, Queue, [&](handler &CGH) {
    AsyncPtrA = static_cast<T *>(
        exp_ext::async_malloc_from_pool(CGH, Size * sizeof(T), MemPool));
  });
  auto AllocNodeB = add_node(Graph, Queue, [&](handler &CGH) {
    AsyncPtrB = static_cast<T *>(
        exp_ext::async_malloc_from_pool(CGH, Size * sizeof(T), MemPool));
  });

  // Assert that we have received unique ptr values, this should always be true
  // regardless of implementation.
  assert((AsyncPtrA != AsyncPtrB));

  // Store pointer values for later comparison
  void *FirstAsyncPtrA = AsyncPtrA;
  void *FirstAsyncPtrB = AsyncPtrB;
  // Add kernel that fills the async allocs with values
  auto KernelFillPtrs = add_node(
      Graph, Queue,
      [&](handler &CGH) {
        depends_on_helper(CGH, {AllocNodeA, AllocNodeB});
        CGH.parallel_for(range<1>(Size), [=](item<1> Item) {
          size_t LinID = Item.get_linear_id();
          // Allocations should be zero initialized on this first use, so we
          // should be able to just add on values rather than set them
          // explicitly.
          AsyncPtrA[LinID] += (1 + LinID);
          AsyncPtrB[LinID] += (2 + LinID);
        });
      },
      AllocNodeA, AllocNodeB);
  // Free all the async allocations, making it possible to reuse them
  auto FreeNodeA = add_node(
      Graph, Queue,
      [&](handler &CGH) {
        depends_on_helper(CGH, KernelFillPtrs);
        exp_ext::async_free(CGH, AsyncPtrA);
      },
      KernelFillPtrs);
  auto FreeNodeB = add_node(
      Graph, Queue,
      [&](handler &CGH) {
        depends_on_helper(CGH, KernelFillPtrs);
        exp_ext::async_free(CGH, AsyncPtrB);
      },
      KernelFillPtrs);

  // Create 2 connected layers in the graph, in each layer we will do a new
  // async_alloc which should reuse one of the 2 previously freed pointers. The
  // other kernel in the layer will simply operate on the output ptr.

  // The first layer will have a direct dependency on a free node, but the
  // subsequent layer will have an indirect dependency. We do not test the order
  // in which allocations are picked for reuse, but we can assume both will be
  // reused by the implementation (same size and properties so they are
  // compatible).

  // First layer, allocation is added first
  auto AllocNodeA1 = add_node(
      Graph, Queue,
      [&](handler &CGH) {
        depends_on_helper(CGH, {FreeNodeA, FreeNodeB});
        AsyncPtrA = static_cast<T *>(
            exp_ext::async_malloc_from_pool(CGH, Size * sizeof(T), MemPool));
      },
      FreeNodeA, FreeNodeB);

  // Check that the new allocation matches one of the previously returned
  // values.
  assert((AsyncPtrA == FirstAsyncPtrA) || (AsyncPtrA == FirstAsyncPtrB));
  // Increment output pointer
  auto KernelInc1 = add_node(
      Graph, Queue,
      [&](handler &CGH) {
        depends_on_helper(CGH, {FreeNodeA, FreeNodeB});
        CGH.parallel_for(range<1>(Size), [=](item<1> Item) {
          size_t LinID = Item.get_linear_id();
          PtrInput[LinID] += 1;
        });
      },
      FreeNodeA, FreeNodeB);

  // Second layer, allocation has an indirect dependency on a free node
  // Increment output pointer
  auto KernelInc2 = add_node(
      Graph, Queue,
      [&](handler &CGH) {
        depends_on_helper(CGH, {KernelInc1});
        CGH.parallel_for(range<1>(Size), [=](item<1> Item) {
          size_t LinID = Item.get_linear_id();
          PtrInput[LinID] += 1;
        });
      },
      KernelInc1);
  auto AllocNodeB2 = add_node(
      Graph, Queue,
      [&](handler &CGH) {
        depends_on_helper(CGH, {KernelInc1, AllocNodeA1});
        AsyncPtrB = static_cast<T *>(
            exp_ext::async_malloc_from_pool(CGH, Size * sizeof(T), MemPool));
      },
      KernelInc1, AllocNodeA1);
  // Check that the new allocation matches one of the previously returned
  // values.
  assert((AsyncPtrB == FirstAsyncPtrA) || (AsyncPtrB == FirstAsyncPtrB));

  // Add a final kernel that adds the async allocation values to the output
  // pointer
  auto KernelAddToOutput = add_node(
      Graph, Queue,
      [&](handler &CGH) {
        depends_on_helper(CGH, {KernelInc2, AllocNodeB2});
        CGH.parallel_for(range<1>(Size), [=](item<1> Item) {
          size_t LinID = Item.get_linear_id();
          PtrInput[LinID] += (AsyncPtrA[LinID] + AsyncPtrB[LinID]);
        });
      },
      KernelInc2, AllocNodeB2);

  // Free the allocations again

  add_node(
      Graph, Queue,
      [&](handler &CGH) {
        depends_on_helper(CGH, KernelAddToOutput);
        exp_ext::async_free(CGH, AsyncPtrA);
      },
      KernelAddToOutput);
  add_node(
      Graph, Queue,
      [&](handler &CGH) {
        depends_on_helper(CGH, KernelAddToOutput);
        exp_ext::async_free(CGH, AsyncPtrB);
      },
      KernelAddToOutput);
}

void calculate_reference_data(size_t Iterations, size_t Size,
                              std::vector<T> &ReferenceOutput) {
  for (size_t i = 0; i < Iterations; i++) {
    for (size_t j = 0; j < Size; j++) {
      ReferenceOutput[j] += 2;
      ReferenceOutput[j] += (j + 1) * (i + 1);
      ReferenceOutput[j] += (j + 2) * (i + 1);
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

  Graph.print_graph("test.dot");

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
