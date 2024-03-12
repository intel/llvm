// This test checks that we can use a stream when explicitly adding a
// command_graph node.

#include "../graph_common.hpp"

int main() {
  queue Queue{};

  using T = int;

  size_t WorkItems = 16;
  std::vector<T> DataIn(WorkItems);

  std::iota(DataIn.begin(), DataIn.end(), 1);

  exp_ext::command_graph Graph{Queue.get_context(), Queue.get_device()};

  T *PtrIn = malloc_device<T>(WorkItems, Queue);
  Queue.copy(DataIn.data(), PtrIn, WorkItems);

  add_node(Graph, Queue, [&](handler &CGH) {
    sycl::stream Out(WorkItems * 16, 16, CGH);
    CGH.parallel_for(range<1>(WorkItems), [=](item<1> id) {
      Out << "Val: " << PtrIn[id.get_linear_id()] << sycl::endl;
    });
  });

  auto GraphExec = Graph.finalize();

  Queue.submit([&](handler &CGH) { CGH.ext_oneapi_graph(GraphExec); });

  Queue.wait_and_throw();

  Queue.copy(PtrIn, DataIn.data(), Size);

  free(PtrIn, Queue);

  return 0;
}
