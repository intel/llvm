// This test uses a host_task when adding a command_graph node.

#include "../graph_common.hpp"

#include <sycl/detail/host_task_impl.hpp>

int main() {
  queue Queue{};

  using T = int;

  const T ModValue = T{7};
  std::vector<T> DataA(Size);

  std::iota(DataA.begin(), DataA.end(), 1000);

  std::vector<T> Reference(DataA);
  for (unsigned n = 0; n < Iterations; n++) {
    for (size_t i = 0; i < Size; i++) {
      Reference[i] += ModValue;
    }
  }

  exp_ext::command_graph Graph{Queue.get_context(), Queue.get_device()};

  T *PtrA = malloc_host<T>(Size, Queue);

  Queue.copy(DataA.data(), PtrA, Size);
  Queue.wait_and_throw();

  // Modify the output values in a host_task
  add_node(Graph, Queue, [&](handler &CGH) {
    CGH.host_task([=]() {
      for (size_t i = 0; i < Size; i++) {
        PtrA[i] += ModValue;
      }
    });
  });

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
  Queue.wait_and_throw();

  free(PtrA, Queue);

  for (size_t i = 0; i < Size; i++) {
    assert(check_value(i, Reference[i], DataA[i], "DataA"));
  }

  return 0;
}
