// Tests destroying finalized command_graph before it is finished executing,
// relying on the backends to properly synchronize and wait for the submitted
// work to finish.

#include "../graph_common.hpp"

int main() {
  queue Queue;

  using T = int;

  std::vector<T> DataA(Size), ReferenceA(Size);

  std::iota(DataA.begin(), DataA.end(), 1);
  std::iota(ReferenceA.begin(), ReferenceA.end(), 2);

  T *PtrA = malloc_device<T>(Size, Queue);

  // Create the command_graph in a seperate scope so that it's destroyed before
  // Queue.wait()
  {
    exp_ext::command_graph Graph{Queue.get_context(), Queue.get_device()};

    Queue.copy(DataA.data(), PtrA, Size);
    Queue.wait_and_throw();

    auto Node = add_node(Graph, Queue, [&](handler &CGH) {
      CGH.parallel_for(Size, [=](item<1> Item) { PtrA[Item.get_id()] += 1; });
    });

    auto GraphExec = Graph.finalize();
    Queue.submit([&](handler &CGH) { CGH.ext_oneapi_graph(GraphExec); });
  }

  Queue.wait_and_throw();

  Queue.copy(PtrA, DataA.data(), Size);
  Queue.wait_and_throw();

  free(PtrA, Queue);

  for (size_t i = 0; i < Size; i++) {
    assert(check_value(i, ReferenceA[i], DataA[i], "DataA"));
  }

  return 0;
}
