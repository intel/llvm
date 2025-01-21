// Tests executable graph update by creating a double buffering scenario, where
// a single graph is repeatedly executed then updated to swap between two sets
// of buffers.

#include "../graph_common.hpp"

int main() {
  queue Queue{};

  using T = int;

  std::vector<T> DataA(Size), DataB(Size), DataC(Size);
  std::vector<T> DataA2(Size), DataB2(Size), DataC2(Size);

  std::iota(DataA.begin(), DataA.end(), 1);
  std::iota(DataB.begin(), DataB.end(), 10);
  std::iota(DataC.begin(), DataC.end(), 1000);

  std::iota(DataA2.begin(), DataA2.end(), 3);
  std::iota(DataB2.begin(), DataB2.end(), 13);
  std::iota(DataC2.begin(), DataC2.end(), 1333);

  std::vector<T> ReferenceA(DataA), ReferenceB(DataB), ReferenceC(DataC);
  std::vector<T> ReferenceA2(DataA2), ReferenceB2(DataB2), ReferenceC2(DataC2);

  calculate_reference_data(Iterations, Size, ReferenceA, ReferenceB,
                           ReferenceC);
  calculate_reference_data(Iterations, Size, ReferenceA2, ReferenceB2,
                           ReferenceC2);

  buffer<T> BufferA{DataA};
  buffer<T> BufferB{DataB};
  buffer<T> BufferC{DataC};

  buffer<T> BufferA2{DataA2};
  buffer<T> BufferB2{DataB2};
  buffer<T> BufferC2{DataC2};

  BufferA.set_write_back(false);
  BufferB.set_write_back(false);
  BufferC.set_write_back(false);
  BufferA2.set_write_back(false);
  BufferB2.set_write_back(false);
  BufferC2.set_write_back(false);

  Queue.wait_and_throw();
  {
    exp_ext::command_graph Graph{
        Queue.get_context(), Queue.get_device(),
        exp_ext::property::graph::assume_buffer_outlives_graph{}};
    add_nodes(Graph, Queue, Size, BufferA, BufferB, BufferC);

    auto ExecGraph = Graph.finalize(exp_ext::property::graph::updatable{});

    // Create second graph using other buffer set
    exp_ext::command_graph GraphUpdate{
        Queue.get_context(), Queue.get_device(),
        exp_ext::property::graph::assume_buffer_outlives_graph{}};
    add_nodes(GraphUpdate, Queue, Size, BufferA2, BufferB2, BufferC2);

    event Event;
    for (size_t i = 0; i < Iterations; i++) {
      Event = Queue.ext_oneapi_graph(ExecGraph);
      // Update to second set of buffers
      ExecGraph.update(GraphUpdate);
      Queue.ext_oneapi_graph(ExecGraph);
      // Reset back to original buffers
      ExecGraph.update(Graph);
    }

    Queue.wait_and_throw();
  }
  host_accessor HostDataA(BufferA);
  host_accessor HostDataB(BufferB);
  host_accessor HostDataC(BufferC);
  host_accessor HostDataA2(BufferA2);
  host_accessor HostDataB2(BufferB2);
  host_accessor HostDataC2(BufferC2);

  for (size_t i = 0; i < Size; i++) {
    assert(check_value(i, ReferenceA[i], HostDataA[i], "DataA"));
    assert(check_value(i, ReferenceB[i], HostDataB[i], "DataB"));
    assert(check_value(i, ReferenceC[i], HostDataC[i], "DataC"));

    assert(check_value(i, ReferenceA2[i], HostDataA2[i], "DataA2"));
    assert(check_value(i, ReferenceB2[i], HostDataB2[i], "DataB2"));
    assert(check_value(i, ReferenceC2[i], HostDataC2[i], "DataC2"));
  }

  return 0;
}
