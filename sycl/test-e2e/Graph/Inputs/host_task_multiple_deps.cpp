// This test uses a host_task with mulitple dependencies
// (before and after the host_task) when adding a command_graph node.

#include "../graph_common.hpp"

int main() {
  queue Queue{};

  using T = int;

  const T ModValue = T{7};
  std::vector<T> DataA(Size), DataB(Size), DataC(Size);

  std::iota(DataA.begin(), DataA.end(), 1);
  std::iota(DataB.begin(), DataB.end(), 10);
  std::iota(DataC.begin(), DataC.end(), 1000);

  std::vector<T> ReferenceA(DataA), ReferenceB(DataB), ReferenceC(DataC);
  for (unsigned n = 0; n < Iterations; n++) {
    for (size_t i = 0; i < Size; i++) {
      ReferenceB[i] = (i + 100);
      ReferenceC[i] = (ReferenceA[i] * (ModValue + (i + 100))) + 1;
      ReferenceA[i] = (i + 100) * (i + 100);
    }
  }

  exp_ext::command_graph Graph{Queue.get_context(), Queue.get_device()};

  T *PtrA = malloc_shared<T>(Size, Queue);
  T *PtrB = malloc_shared<T>(Size, Queue);
  T *PtrC = malloc_shared<T>(Size, Queue);

  Queue.copy(DataA.data(), PtrA, Size);
  Queue.copy(DataB.data(), PtrB, Size);
  Queue.copy(DataC.data(), PtrC, Size);
  Queue.wait_and_throw();

  // Vector add to output
  auto NodeA = add_node(Graph, Queue, [&](handler &CGH) {
    CGH.parallel_for(range<1>(Size), [=](item<1> id) { PtrC[id] = PtrA[id]; });
  });

  // Vector add to output
  auto NodeB = add_node(Graph, Queue, [&](handler &CGH) {
    CGH.parallel_for(range<1>(Size), [=](item<1> id) { PtrB[id] = 100 + id; });
  });

  // Modify the output values in a host_task
  auto NodeC = add_node(
      Graph, Queue,
      [&](handler &CGH) {
        depends_on_helper(CGH, {NodeA, NodeB});
        CGH.host_task([=]() {
          for (size_t i = 0; i < Size; i++) {
            PtrC[i] *= (ModValue + PtrB[i]);
            PtrA[i] = PtrB[i];
          }
        });
      },
      NodeA, NodeB);

  // Modify temp buffer and write to output buffer
  add_node(
      Graph, Queue,
      [&](handler &CGH) {
        depends_on_helper(CGH, NodeC);
        CGH.parallel_for(range<1>(Size), [=](item<1> id) { PtrC[id] += 1; });
      },
      NodeC);

  // Modify temp buffer and write to output buffer
  add_node(
      Graph, Queue,
      [&](handler &CGH) {
        depends_on_helper(CGH, NodeC);
        CGH.parallel_for(range<1>(Size),
                         [=](item<1> id) { PtrA[id] *= PtrB[id]; });
      },
      NodeC);

  auto GraphExec = Graph.finalize();

  event Event;
  for (unsigned n = 0; n < Iterations; n++) {
    Event = Queue.submit([&](handler &CGH) {
      CGH.depends_on(Event);
      CGH.ext_oneapi_graph(GraphExec);
    });
    Event.wait();
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
