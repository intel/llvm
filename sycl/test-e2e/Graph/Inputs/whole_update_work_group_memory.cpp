// Tests whole graph update of nodes with the work_group_memory extension

#include "../graph_common.hpp"
#include <sycl/ext/oneapi/experimental/work_group_memory.hpp>

auto add_graph_node(
    exp_ext::command_graph<exp_ext::graph_state::modifiable> &Graph,
    queue &Queue, size_t LocalSize, int *Ptr) {
  return add_node(Graph, Queue, [&](handler &CGH) {
    exp_ext::work_group_memory<int[]> WGMem{LocalSize, CGH};

    CGH.parallel_for(nd_range({Size}, {LocalSize}), [=](nd_item<1> Item) {
      WGMem[Item.get_local_linear_id()] = Item.get_global_linear_id() * 2;
      Ptr[Item.get_global_linear_id()] +=
          WGMem[Item.get_local_linear_id()] + Item.get_local_range(0);
    });
  });
}

int main() {
  queue Queue{};

  const size_t LocalSize = 128;

  std::vector<int> DataA(Size), DataB(Size);

  std::iota(DataA.begin(), DataA.end(), 10);
  std::iota(DataB.begin(), DataB.end(), 10);

  exp_ext::command_graph GraphA{Queue.get_context(), Queue.get_device()};

  int *PtrA = malloc_device<int>(Size, Queue);
  int *PtrB = malloc_device<int>(Size, Queue);

  Queue.copy(DataA.data(), PtrA, Size);
  Queue.copy(DataB.data(), PtrB, Size);
  Queue.wait_and_throw();

  auto NodeA = add_graph_node(GraphA, Queue, LocalSize / 2, PtrA);

  auto GraphExecA = GraphA.finalize(exp_ext::property::graph::updatable{});

  // Create second graph for whole graph update with a different local size
  exp_ext::command_graph GraphB{Queue.get_context(), Queue.get_device()};
  auto NodeB = add_graph_node(GraphB, Queue, LocalSize, PtrB);

  // Execute graphs before updating and check outputs
  for (unsigned n = 0; n < Iterations; n++) {
    Queue.submit([&](handler &CGH) { CGH.ext_oneapi_graph(GraphExecA); });
  }

  Queue.wait_and_throw();

  Queue.copy(PtrA, DataA.data(), Size);
  Queue.copy(PtrB, DataB.data(), Size);
  Queue.wait_and_throw();

  for (size_t i = 0; i < Size; i++) {
    int RefA = 10 + i + Iterations * ((i * 2) + (LocalSize / 2));
    int RefB = 10 + i;
    assert(check_value(i, RefA, DataA[i], "PtrA"));
    assert(check_value(i, RefB, DataB[i], "PtrB"));
  }

  // Update GraphExecA using whole graph update

  GraphExecA.update(GraphB);

  // Execute graphs again and check outputs
  for (unsigned N = 0; N < Iterations; N++) {
    Queue.submit([&](handler &CGH) { CGH.ext_oneapi_graph(GraphExecA); });
  }

  Queue.wait_and_throw();

  Queue.copy(PtrA, DataA.data(), Size);
  Queue.copy(PtrB, DataB.data(), Size);
  Queue.wait_and_throw();

  for (size_t i = 0; i < Size; i++) {
    int RefA = 10 + i + Iterations * ((i * 2) + (LocalSize / 2));
    int RefB = 10 + i + Iterations * ((i * 2) + LocalSize);
    assert(check_value(i, RefA, DataA[i], "PtrA"));
    assert(check_value(i, RefB, DataB[i], "PtrB"));
  }

  free(PtrA, Queue);
  free(PtrB, Queue);
  return 0;
}
