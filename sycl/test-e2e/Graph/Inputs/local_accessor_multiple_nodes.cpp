// Test creating a graph where more than one nodes uses local accessors,
// and submits of the graph.

#include "../graph_common.hpp"

int main() {
  queue Queue{};

  using T = int;

  const size_t LocalSize = 128;

  std::vector<T> HostData(Size);

  std::iota(HostData.begin(), HostData.end(), 10);

  exp_ext::command_graph Graph{Queue.get_context(), Queue.get_device()};

  T *PtrA = malloc_device<T>(Size, Queue);

  Queue.copy(HostData.data(), PtrA, Size);
  Queue.wait_and_throw();

  auto NodeA = add_node(Graph, Queue, [&](handler &CGH) {
    local_accessor<T, 1> LocalMem(LocalSize, CGH);

    CGH.parallel_for(nd_range({Size}, {LocalSize}), [=](nd_item<1> Item) {
      LocalMem[Item.get_local_linear_id()] = Item.get_global_linear_id() * 2;
      PtrA[Item.get_global_linear_id()] += LocalMem[Item.get_local_linear_id()];
    });
  });

  auto NodeB = add_node(
      Graph, Queue,
      [&](handler &CGH) {
        local_accessor<T, 1> LocalMem(LocalSize, CGH);
        depends_on_helper(CGH, NodeA);

        CGH.parallel_for(nd_range({Size}, {LocalSize}), [=](nd_item<1> Item) {
          LocalMem[Item.get_local_linear_id()] = 3;
          PtrA[Item.get_global_linear_id()] *=
              LocalMem[Item.get_local_linear_id()];
        });
      },
      NodeA);

  auto GraphExec = Graph.finalize();

  for (unsigned n = 0; n < Iterations; n++) {
    Queue.submit([&](handler &CGH) { CGH.ext_oneapi_graph(GraphExec); });
  }

  Queue.wait_and_throw();

  Queue.copy(PtrA, HostData.data(), Size);
  Queue.wait_and_throw();

  free(PtrA, Queue);

  for (size_t i = 0; i < Size; i++) {
    T Ref = 10 + i;

    for (size_t n = 0; n < Iterations; n++) {
      Ref += i * 2;
      Ref *= 3;
    }
    assert(check_value(i, Ref, HostData[i], "PtrA"));
  }

  return 0;
}
