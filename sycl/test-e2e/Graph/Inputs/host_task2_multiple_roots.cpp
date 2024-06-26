// This test uses a host_task when adding a command_graph node for graph with
// multiple roots.

#include "../graph_common.hpp"

#include <sycl/detail/host_task_impl.hpp>

int main() {
  queue Queue{};

  using T = int;

  const T ModValue = T{7};
  std::vector<T> DataA(Size), DataB(Size), DataC(Size), Res2(Size);

  std::iota(DataA.begin(), DataA.end(), 1);
  std::iota(DataB.begin(), DataB.end(), 10);
  std::iota(DataC.begin(), DataC.end(), 1000);

  std::vector<T> Reference(DataC);
  for (unsigned n = 0; n < Iterations; n++) {
    for (size_t i = 0; i < Size; i++) {
      Reference[i] = (((DataA[i] + DataB[i]) * ModValue) + 1) * DataB[i];
    }
  }

  exp_ext::command_graph Graph{Queue.get_context(), Queue.get_device()};

  T *PtrA = malloc_device<T>(Size, Queue);
  T *PtrB = malloc_device<T>(Size, Queue);
  T *PtrC = malloc_shared<T>(Size, Queue);
  T *PtrA2 = malloc_device<T>(Size, Queue);
  T *PtrB2 = malloc_device<T>(Size, Queue);
  T *PtrC2 = malloc_shared<T>(Size, Queue);

  Queue.copy(DataA.data(), PtrA, Size);
  Queue.copy(DataB.data(), PtrB, Size);
  Queue.copy(DataC.data(), PtrC, Size);
  Queue.copy(DataA.data(), PtrA2, Size);
  Queue.copy(DataB.data(), PtrB2, Size);
  Queue.copy(DataC.data(), PtrC2, Size);
  Queue.wait_and_throw();

  // Vector add to output
  auto NodeA = add_node(Graph, Queue, [&](handler &CGH) {
    CGH.parallel_for(range<1>(Size), [=](item<1> id) { PtrC[id] = PtrA[id]; });
  });

  // Vector add to output
  auto NodeB = add_node(
      Graph, Queue,
      [&](handler &CGH) {
        depends_on_helper(CGH, NodeA);
        CGH.parallel_for(range<1>(Size),
                         [=](item<1> id) { PtrC[id] += PtrB[id]; });
      },
      NodeA);

  // Modify the output values in a host_task
  auto NodeC = add_node(
      Graph, Queue,
      [&](handler &CGH) {
        depends_on_helper(CGH, NodeB);
        CGH.host_task([=]() {
          for (size_t i = 0; i < Size; i++) {
            PtrC[i] *= ModValue;
          }
        });
      },
      NodeB);

  // Modify temp buffer and write to output buffer
  auto NodeD = add_node(
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
        depends_on_helper(CGH, NodeD);
        CGH.parallel_for(range<1>(Size),
                         [=](item<1> id) { PtrC[id] *= PtrB[id]; });
      },
      NodeD);

  // Vector add to output
  auto NodeA2 = add_node(Graph, Queue, [&](handler &CGH) {
    CGH.parallel_for(range<1>(Size),
                     [=](item<1> id) { PtrC2[id] = PtrA2[id]; });
  });

  // Vector add to output
  auto NodeB2 = add_node(
      Graph, Queue,
      [&](handler &CGH) {
        depends_on_helper(CGH, NodeA2);
        CGH.parallel_for(range<1>(Size),
                         [=](item<1> id) { PtrC2[id] += PtrB2[id]; });
      },
      NodeA2);

  // Modify the output values in a host_task
  auto NodeC2 = add_node(
      Graph, Queue,
      [&](handler &CGH) {
        depends_on_helper(CGH, NodeB2);
        CGH.host_task([=]() {
          for (size_t i = 0; i < Size; i++) {
            PtrC2[i] *= ModValue;
          }
        });
      },
      NodeB2);

  // Modify temp buffer and write to output buffer
  auto NodeD2 = add_node(
      Graph, Queue,
      [&](handler &CGH) {
        depends_on_helper(CGH, NodeC2);
        CGH.parallel_for(range<1>(Size), [=](item<1> id) { PtrC2[id] += 1; });
      },
      NodeC2);

  // Modify temp buffer and write to output buffer
  add_node(
      Graph, Queue,
      [&](handler &CGH) {
        depends_on_helper(CGH, NodeD2);
        CGH.parallel_for(range<1>(Size),
                         [=](item<1> id) { PtrC2[id] *= PtrB2[id]; });
      },
      NodeD2);

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
  Queue.copy(PtrC2, Res2.data(), Size);
  Queue.wait_and_throw();

  free(PtrA, Queue);
  free(PtrB, Queue);
  free(PtrC, Queue);
  free(PtrA2, Queue);
  free(PtrB2, Queue);
  free(PtrC2, Queue);

  for (size_t i = 0; i < Size; i++) {
    assert(check_value(i, Reference[i], DataC[i], "DataC"));
    assert(check_value(i, Reference[i], Res2[i], "Res2"));
  }

  return 0;
}
