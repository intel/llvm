// This test uses a host_task as a last node

#include "../graph_common.hpp"

#include <sycl/detail/host_task_impl.hpp>

int main() {
  queue Queue{};

  using T = int;

  const T ModValue = T{7};
  std::vector<T> DataA(Size), DataB(Size), DataC(Size);

  std::iota(DataA.begin(), DataA.end(), 1);
  std::iota(DataB.begin(), DataB.end(), 10);
  std::iota(DataC.begin(), DataC.end(), 1000);

  std::vector<T> Reference(DataC);
  for (unsigned n = 0; n < Iterations; n++) {
    for (size_t i = 0; i < Size; i++) {
      Reference[i] += (DataA[i] + DataB[i]) + ModValue;
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
    CGH.parallel_for(range<1>(Size),
                     [=](item<1> id) { PtrC[id] += PtrA[id] + PtrB[id]; });
  });

  // Modify the output values in a host_task
  auto NodeB = add_node(
      Graph, Queue,
      [&](handler &CGH) {
        depends_on_helper(CGH, NodeA);
        CGH.host_task([=]() {
          for (size_t i = 0; i < Size; i++) {
            PtrC[i] += ModValue;
          }
        });
      },
      NodeA);

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
