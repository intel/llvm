// Tests whole graph update with a host-task in the graph

#include "../graph_common.hpp"

using T = int;

void calculate_reference_data(size_t Iterations, size_t Size,
                              std::vector<T> &ReferenceA,
                              std::vector<T> &ReferenceB,
                              std::vector<T> &ReferenceC, T ModValue) {
  for (size_t n = 0; n < Iterations; n++) {
    for (size_t i = 0; i < Size; i++) {
      ReferenceA[i]++;
      ReferenceB[i] += ReferenceA[i];
      ReferenceC[i] -= ReferenceA[i];
      ReferenceB[i]--;
      ReferenceC[i]--;
      ReferenceC[i] += ModValue;
      ReferenceC[i] *= 2;
    }
  }
}

void add_nodes_to_graph(
    exp_ext::command_graph<exp_ext::graph_state::modifiable> &Graph,
    queue &Queue, T *PtrA, T *PtrB, T *PtrC, T ModValue) {
  // Add some commands to the graph
  auto LastOperation = add_nodes(Graph, Queue, Size, PtrA, PtrB, PtrC);

  // Add a host task which modifies PtrC
  auto HostTaskOp = add_node(
      Graph, Queue,
      [&](handler &CGH) {
        depends_on_helper(CGH, LastOperation);
        CGH.host_task([=]() {
          for (size_t i = 0; i < Size; i++) {
            PtrC[i] += ModValue;
          }
        });
      },
      LastOperation);

  // Add another node that depends on the host-task
  add_node(
      Graph, Queue,
      [&](handler &CGH) {
        depends_on_helper(CGH, HostTaskOp);
        CGH.parallel_for(range<1>(Size), [=](item<1> Item) {
          PtrC[Item.get_linear_id()] *= 2;
        });
      },
      HostTaskOp);
}

int main() {
  queue Queue{};

  std::vector<T> DataA(Size), DataB(Size), DataC(Size);

  std::iota(DataA.begin(), DataA.end(), 1);
  std::iota(DataB.begin(), DataB.end(), 10);
  std::iota(DataC.begin(), DataC.end(), 1000);

  auto DataA2 = DataA;
  auto DataB2 = DataB;
  auto DataC2 = DataC;

  const T ModValue = 7;
  std::vector<T> ReferenceA(DataA), ReferenceB(DataB), ReferenceC(DataC);
  calculate_reference_data(Iterations, Size, ReferenceA, ReferenceB, ReferenceC,
                           ModValue);

  exp_ext::command_graph GraphA{Queue.get_context(), Queue.get_device()};

  T *PtrA = malloc_device<T>(Size, Queue);
  T *PtrB = malloc_device<T>(Size, Queue);
  T *PtrC = malloc_shared<T>(Size, Queue);

  Queue.copy(DataA.data(), PtrA, Size);
  Queue.copy(DataB.data(), PtrB, Size);
  Queue.copy(DataC.data(), PtrC, Size);
  Queue.wait_and_throw();

  // Fill graphA with nodes
  add_nodes_to_graph(GraphA, Queue, PtrA, PtrB, PtrC, ModValue);

  auto GraphExec = GraphA.finalize(exp_ext::property::graph::updatable{});

  exp_ext::command_graph GraphB{Queue.get_context(), Queue.get_device()};

  T *PtrA2 = malloc_device<T>(Size, Queue);
  T *PtrB2 = malloc_device<T>(Size, Queue);
  T *PtrC2 = malloc_shared<T>(Size, Queue);

  Queue.copy(DataA2.data(), PtrA2, Size);
  Queue.copy(DataB2.data(), PtrB2, Size);
  Queue.copy(DataC2.data(), PtrC2, Size);
  Queue.wait_and_throw();

  // Fill graphB with nodes, with a different set of pointers
  add_nodes_to_graph(GraphB, Queue, PtrA2, PtrB2, PtrC2, ModValue);

  // Execute several Iterations of the graph, updating in between each
  // execution.
  event Event;
  for (unsigned n = 0; n < Iterations; n++) {
    Event = Queue.submit([&](handler &CGH) {
      CGH.depends_on(Event);
      CGH.ext_oneapi_graph(GraphExec);
    });
    GraphExec.update(GraphB);
    Event = Queue.submit([&](handler &CGH) {
      CGH.depends_on(Event);
      CGH.ext_oneapi_graph(GraphExec);
    });
    GraphExec.update(GraphA);
  }

  Queue.wait_and_throw();

  Queue.copy(PtrA, DataA.data(), Size);
  Queue.copy(PtrB, DataB.data(), Size);
  Queue.copy(PtrC, DataC.data(), Size);

  Queue.copy(PtrA2, DataA2.data(), Size);
  Queue.copy(PtrB2, DataB2.data(), Size);
  Queue.copy(PtrC2, DataC2.data(), Size);
  Queue.wait_and_throw();

  free(PtrA, Queue);
  free(PtrB, Queue);
  free(PtrC, Queue);

  free(PtrA2, Queue);
  free(PtrB2, Queue);
  free(PtrC2, Queue);

  for (size_t i = 0; i < Size; i++) {
    assert(check_value(i, ReferenceA[i], DataA[i], "DataA"));
    assert(check_value(i, ReferenceB[i], DataB[i], "DataB"));
    assert(check_value(i, ReferenceC[i], DataC[i], "DataC"));

    assert(check_value(i, ReferenceA[i], DataA2[i], "DataA2"));
    assert(check_value(i, ReferenceB[i], DataB2[i], "DataB2"));
    assert(check_value(i, ReferenceC[i], DataC2[i], "DataC2"));
  }

  return 0;
}
