// Tests queue shortcuts for executing a graph.

#include "../graph_common.hpp"

int main() {
  queue Queue{{sycl::ext::intel::property::queue::no_immediate_command_list{}}};

  if (!are_graphs_supported(Queue)) {
    return 0;
  }

  using T = int;

  std::vector<T> DataA(Size), DataB(Size), DataC(Size);

  std::iota(DataA.begin(), DataA.end(), 1);
  std::iota(DataB.begin(), DataB.end(), 10);
  std::iota(DataC.begin(), DataC.end(), 1000);

  std::vector<T> ReferenceA(DataA), ReferenceB(DataB), ReferenceC(DataC);
  calculate_reference_data(Iterations, Size, ReferenceA, ReferenceB,
                           ReferenceC);

  exp_ext::command_graph Graph{Queue.get_context(), Queue.get_device()};

  buffer<T> BufferA{DataA.data(), range<1>{DataA.size()}};
  buffer<T> BufferB{DataB.data(), range<1>{DataB.size()}};
  buffer<T> BufferC{DataC.data(), range<1>{DataC.size()}};

  T *PtrA = malloc_device<T>(Size, Queue);
  T *PtrB = malloc_device<T>(Size, Queue);
  T *PtrC = malloc_device<T>(Size, Queue);

  Queue.copy(DataA.data(), PtrA, Size);
  Queue.copy(DataB.data(), PtrB, Size);
  Queue.copy(DataC.data(), PtrC, Size);
  Queue.wait_and_throw();

  add_nodes(Graph, Queue, Size, PtrA, PtrB, PtrC);

  auto GraphExec = Graph.finalize();

  // Execute several iterations of the graph using the different shortcuts
  event Event = Queue.ext_oneapi_graph(GraphExec);

  assert(Iterations > 2);
  const size_t LoopIterations = Iterations - 2;
  std::vector<event> Events(LoopIterations);
  for (unsigned n = 0; n < LoopIterations; n++) {
    Events[n] = Queue.ext_oneapi_graph(GraphExec, Event);
  }

  Queue.ext_oneapi_graph(GraphExec, Events).wait();

  Queue.copy(PtrA, DataA.data(), Size);
  Queue.copy(PtrB, DataB.data(), Size);
  Queue.copy(PtrC, DataC.data(), Size);
  Queue.wait_and_throw();

  free(PtrA, Queue);
  free(PtrB, Queue);
  free(PtrC, Queue);

  for (size_t i = 0; i < Size; i++) {
    assert(check_value(i, ReferenceA[i], DataA[i], "DataA"));
    assert(check_value(i, ReferenceB[i], DataB[i], "DataB"));
    assert(check_value(i, ReferenceC[i], DataC[i], "DataC"));
  }

  return 0;
}
