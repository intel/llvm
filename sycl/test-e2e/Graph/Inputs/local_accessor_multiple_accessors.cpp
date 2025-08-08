// Tests adding of nodes with more than one local accessor,
// and submitting the graph.

#include "../graph_common.hpp"

int main() {
  queue Queue{};

  using T = int;

  const size_t LocalSize = 128;

  std::vector<T> HostData(Size);

  std::iota(HostData.begin(), HostData.end(), 10);

  exp_ext::command_graph Graph{Queue.get_context(), Queue.get_device()};

  T *PtrIn = malloc_device<T>(Size, Queue);
  T *PtrOut = malloc_device<T>(Size, Queue);

  Queue.memset(PtrOut, 0, Size * sizeof(T));
  Queue.copy(HostData.data(), PtrIn, Size);
  Queue.wait_and_throw();

  auto Node = add_node(Graph, Queue, [&](handler &CGH) {
    local_accessor<T, 1> LocalMemA(LocalSize, CGH);
    local_accessor<T, 1> LocalMemB(LocalSize, CGH);

    CGH.parallel_for(nd_range({Size}, {LocalSize}), [=](nd_item<1> Item) {
      auto LocalID = Item.get_local_linear_id();
      auto GlobalID = Item.get_global_linear_id();
      LocalMemA[LocalID] = GlobalID;
      LocalMemB[LocalID] = PtrIn[GlobalID];
      PtrOut[GlobalID] += LocalMemA[LocalID] * LocalMemB[LocalID];
    });
  });

  auto GraphExec = Graph.finalize();

  for (unsigned n = 0; n < Iterations; n++) {
    Queue.submit([&](handler &CGH) { CGH.ext_oneapi_graph(GraphExec); });
  }

  Queue.wait_and_throw();

  Queue.copy(PtrOut, HostData.data(), Size);
  Queue.wait_and_throw();

  free(PtrIn, Queue);
  free(PtrOut, Queue);

  for (size_t i = 0; i < Size; i++) {
    T Ref = 0;
    for (size_t n = 0; n < Iterations; n++) {
      Ref += (i * (10 + i));
    }
    assert(check_value(i, Ref, HostData[i], "PtrOut"));
  }

  return 0;
}
