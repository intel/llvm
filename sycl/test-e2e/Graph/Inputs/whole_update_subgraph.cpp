// Tests that whole graph update works when using sub-graphs.

#include "../graph_common.hpp"

template <typename T>
void constructGraphs(
    queue Queue, exp_ext::command_graph<exp_ext::graph_state::modifiable> Graph,
    exp_ext::command_graph<exp_ext::graph_state::modifiable> SubGraph,
    T *Data) {

  add_node(SubGraph, Queue, [&](handler &CGH) {
    CGH.parallel_for(range<1>(Size), [=](item<1> Id) { Data[Id] += 2; });
  });

  exp_ext::command_graph SubGraphExec = SubGraph.finalize();

  auto NodeA = add_node(Graph, Queue, [&](handler &CGH) {
    CGH.parallel_for(range<1>(Size), [=](item<1> Id) { Data[Id] += 1; });
  });

  auto NodeB = add_node(Graph, Queue, [&](handler &CGH) {
    depends_on_helper(CGH, NodeA);
    CGH.ext_oneapi_graph(SubGraphExec);
  });
}

int main() {
  queue Queue{};

  using T = int;

  std::vector<T> DataHost(Size, 1);
  std::vector<T> DataHostUpdate(Size, 1);
  T *DataDevice = malloc_device<T>(Size, Queue);
  T *DataDeviceUpdate = malloc_device<T>(Size, Queue);
  Queue.copy(DataHost.data(), DataDevice, Size);
  Queue.copy(DataHost.data(), DataDeviceUpdate, Size);

  exp_ext::command_graph SubGraphA{Queue.get_context(), Queue.get_device()};
  exp_ext::command_graph SubGraphB{Queue.get_context(), Queue.get_device()};

  exp_ext::command_graph GraphA{Queue.get_context(), Queue.get_device()};
  exp_ext::command_graph GraphB{Queue.get_context(), Queue.get_device()};

  constructGraphs<T>(Queue, GraphA, SubGraphA, DataDevice);
  auto GraphExec = GraphA.finalize(exp_ext::property::graph::updatable{});
  Queue.ext_oneapi_graph(GraphExec).wait();

  constructGraphs<T>(Queue, GraphB, SubGraphB, DataDeviceUpdate);

  bool GotException = false;
  try {
    GraphExec.update(GraphB);
  } catch (sycl::exception &e) {
    // TODO The subgraph update feature is not implemented yet. For now this
    // is the expected behaviour.
    return 0;
  }
  assert(!GotException);

  Queue.ext_oneapi_graph(GraphExec).wait();

  Queue.copy(DataDevice, DataHost.data(), Size);
  Queue.copy(DataDeviceUpdate, DataHostUpdate.data(), Size);
  Queue.wait_and_throw();

  for (size_t i = 0; i < Size; i++) {
    assert(check_value(i, 4, DataHost[i], "DataHost"));
    assert(check_value(i, 4, DataHostUpdate[i], "DataHostUpdate"));
  }

  return 0;
}
