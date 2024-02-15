// Tests basic adding of nodes with USM pointers,
// and submission of the graph.

#include "../graph_common.hpp"
#include <thread>

int main() {
  queue Queue{};

  if (!are_graphs_supported(Queue)) {
    return 0;
  }

  using T = int;

  const unsigned NumThreads = std::thread::hardware_concurrency();
  std::vector<T> DataA(Size), DataB(Size), DataC(Size);

  std::iota(DataA.begin(), DataA.end(), 1);
  std::iota(DataB.begin(), DataB.end(), 10);
  std::iota(DataC.begin(), DataC.end(), 1000);

  std::vector<T> ReferenceA(DataA), ReferenceB(DataB), ReferenceC(DataC);
  calculate_reference_data(Iterations, Size, ReferenceA, ReferenceB,
                           ReferenceC);

  exp_ext::command_graph Graph{Queue.get_context(), Queue.get_device()};

  T *PtrA = malloc_device<T>(Size, Queue);
  T *PtrB = malloc_device<T>(Size, Queue);
  T *PtrC = malloc_device<T>(Size, Queue);

  Queue.copy(DataA.data(), PtrA, Size);
  Queue.copy(DataB.data(), PtrB, Size);
  Queue.copy(DataC.data(), PtrC, Size);
  Queue.wait_and_throw();

  // Add commands to graph
  add_nodes(Graph, Queue, Size, PtrA, PtrB, PtrC);

  Barrier SyncPoint{NumThreads};

  auto GraphExec = Graph.finalize();

  auto SubmitGraph = [&]() {
    SyncPoint.wait();
    Queue.submit([&](handler &CGH) { CGH.ext_oneapi_graph(GraphExec); });
  };

  event Event;
  for (unsigned n = 0; n < Iterations; n++) {
    Event =
        Queue.submit([&](handler &CGH) { CGH.ext_oneapi_graph(GraphExec); });
  }

  Queue.wait_and_throw();

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
