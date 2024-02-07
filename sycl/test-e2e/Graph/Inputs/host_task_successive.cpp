// This test uses a host_task when adding a command_graph node.

#include "../graph_common.hpp"

int main() {
  queue Queue{};

  if (!are_graphs_supported(Queue)) {
    return 0;
  }

  using T = int;

  if (!Queue.get_device().has(sycl::aspect::usm_shared_allocations)) {
    return 0;
  }

  const T ModValue = T{7};
  std::vector<T> DataA(Size), DataB(Size), DataC(Size);

  std::iota(DataA.begin(), DataA.end(), 1);
  std::iota(DataB.begin(), DataB.end(), 10);
  std::iota(DataC.begin(), DataC.end(), 1000);

  std::vector<T> Reference(DataC);
  for (unsigned n = 0; n < Iterations; n++) {
    for (size_t i = 0; i < Size; i++) {
      Reference[i] = ((DataA[i] * ModValue) + 10) * 2;
    }
  }

  exp_ext::command_graph Graph{Queue.get_context(), Queue.get_device()};

  T *PtrA = malloc_device<T>(Size, Queue);
  T *PtrB = malloc_device<T>(Size, Queue);
  T *PtrC = malloc_shared<T>(Size, Queue);

  Queue.copy(DataA.data(), PtrA, Size);
  Queue.copy(DataB.data(), PtrB, Size);
  Queue.copy(DataC.data(), PtrC, Size);
  Queue.wait_and_throw();

  // Vector add to output
  auto NodeA = add_node(Graph, Queue, [&](handler &CGH) {
    CGH.parallel_for(range<1>(Size), [=](item<1> id) { PtrC[id] = PtrA[id]; });
  });

  // Modify the output values in a host_task
  auto NodeB = add_node(
      Graph, Queue,
      [&](handler &CGH) {
        depends_on_helper(CGH, NodeA);
        CGH.host_task([=]() {
          for (size_t i = 0; i < Size; i++) {
            PtrC[i] *= ModValue;
          }
        });
      },
      NodeA);

  // Modify the output values in a host_task
  auto NodeC = add_node(
      Graph, Queue,
      [&](handler &CGH) {
        depends_on_helper(CGH, NodeB);
        CGH.host_task([=]() {
          for (size_t i = 0; i < Size; i++) {
            PtrC[i] += 10;
          }
        });
      },
      NodeB);

  // Modify temp buffer and write to output buffer
  add_node(
      Graph, Queue,
      [&](handler &CGH) {
        depends_on_helper(CGH, NodeC);
        CGH.parallel_for(range<1>(Size), [=](item<1> id) { PtrC[id] *= 2; });
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

  Queue.copy(PtrC, DataC.data(), Size);
  Queue.wait_and_throw();

  free(PtrA, Queue);
  free(PtrB, Queue);
  free(PtrC, Queue);

  for (size_t i = 0; i < Size; i++) {
    assert(check_value(i, Reference[i], DataC[i], "DataC"));
  }

  return 0;
}
