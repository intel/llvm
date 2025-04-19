// Tests basic adding of async allocation nodes for device allocations

#include "../../graph_common.hpp"
#include <sycl/ext/oneapi/experimental/async_alloc/async_alloc.hpp>

using T = int;
void add_nodes_to_graph(
    exp_ext::command_graph<exp_ext::graph_state::modifiable> &Graph,
    queue &Queue, size_t Size, T *PtrInput) {
  T *AsyncPtrA = nullptr;
  // Add alloc node
  auto AllocNode = add_node(Graph, Queue, [&](handler &CGH) {
    AsyncPtrA = static_cast<T *>(
        exp_ext::async_malloc(CGH, usm::alloc::device, Size * sizeof(T)));
  });
  // Add memcpy to alloc node
  auto MemcpyNodeA = add_node(
      Graph, Queue,
      [&](handler &CGH) {
        depends_on_helper(CGH, AllocNode);
        CGH.memcpy(AsyncPtrA, PtrInput, Size * sizeof(T));
      },
      AllocNode);

  // add kernel that operates on async memory
  auto KernelNodeA = add_node(
      Graph, Queue,
      [&](handler &CGH) {
        depends_on_helper(CGH, MemcpyNodeA);
        CGH.parallel_for(range<1>(Size), [=](item<1> Item) {
          size_t LinID = Item.get_linear_id();
          PtrInput[LinID] += AsyncPtrA[LinID];
        });
      },
      MemcpyNodeA);

  // Add free node

  auto FreeNode = add_node(
      Graph, Queue,
      [&](handler &CGH) {
        depends_on_helper(CGH, KernelNodeA);
        exp_ext::async_free(CGH, AsyncPtrA);
      },
      KernelNodeA);

  // Add alloc node

  T *AsyncPtrB = nullptr;

  auto AllocNodeB = add_node(
      Graph, Queue,
      [&](handler &CGH) {
        depends_on_helper(CGH, FreeNode);
        AsyncPtrB = static_cast<T *>(
            exp_ext::async_malloc(CGH, usm::alloc::device, Size * sizeof(T)));
      },
      FreeNode);

  // Add kernels that operates on async memory
  auto KernelNodeB = add_node(
      Graph, Queue,
      [&](handler &CGH) {
        depends_on_helper(CGH, AllocNodeB);
        CGH.parallel_for(range<1>(Size), [=](item<1> Item) {
          size_t LinID = Item.get_linear_id();
          AsyncPtrB[LinID] = PtrInput[LinID] + LinID;
        });
      },
      AllocNodeB);
  auto KernelNodeC = add_node(
      Graph, Queue,
      [&](handler &CGH) {
        depends_on_helper(CGH, KernelNodeB);
        CGH.parallel_for(range<1>(Size), [=](item<1> Item) {
          size_t LinID = Item.get_linear_id();
          AsyncPtrB[LinID] *= 3;
        });
      },
      KernelNodeB);
  // Add copy back to input USM pointer
  auto MemcpyNodeB = add_node(
      Graph, Queue,
      [&](handler &CGH) {
        depends_on_helper(CGH, KernelNodeC);
        CGH.memcpy(PtrInput, AsyncPtrB, Size * sizeof(T));
      },
      KernelNodeC);
  // Add free node
  add_node(
      Graph, Queue,
      [&](handler &CGH) {
        depends_on_helper(CGH, MemcpyNodeB);
        exp_ext::async_free(CGH, AsyncPtrB);
      },
      MemcpyNodeB);
}

void calculate_reference_data(size_t Iterations, size_t Size,
                              std::vector<T> &ReferenceOutput) {
  for (size_t i = 0; i < Iterations; i++) {
    for (size_t j = 0; j < Size; j++) {
      ReferenceOutput[j] *= 2;
      ReferenceOutput[j] += j;
      ReferenceOutput[j] *= 3;
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
