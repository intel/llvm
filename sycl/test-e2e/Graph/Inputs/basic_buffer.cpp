// Tests adding nodes to a graph and submitting the graph
// using buffers accessors for inputs and outputs.

#include "../graph_common.hpp"

int main() {
  queue Queue{};

  using T = unsigned short;

  std::vector<T> DataA(Size), DataB(Size), DataC(Size);

  std::iota(DataA.begin(), DataA.end(), 1);
  std::iota(DataB.begin(), DataB.end(), 10);
  std::iota(DataC.begin(), DataC.end(), 1000);

  std::vector<T> ReferenceA(DataA), ReferenceB(DataB), ReferenceC(DataC);
  calculate_reference_data(Iterations, Size, ReferenceA, ReferenceB,
                           ReferenceC);

  buffer<T> BufferA{DataA.data(), range<1>{DataA.size()}};
  BufferA.set_write_back(false);
  buffer<T> BufferB{DataB.data(), range<1>{DataB.size()}};
  BufferB.set_write_back(false);
  buffer<T> BufferC{DataC.data(), range<1>{DataC.size()}};
  BufferC.set_write_back(false);
  {
    exp_ext::command_graph Graph{
        Queue.get_context(),
        Queue.get_device(),
        {exp_ext::property::graph::assume_buffer_outlives_graph{}}};

    // Add commands to graph
    add_nodes(Graph, Queue, Size, BufferA, BufferB, BufferC);

    auto GraphExec = Graph.finalize();

    for (unsigned n = 0; n < Iterations; n++) {
      Queue.submit([&](handler &CGH) { CGH.ext_oneapi_graph(GraphExec); });
    }
    Queue.wait_and_throw();
  }

  host_accessor HostAccA(BufferA);
  host_accessor HostAccB(BufferB);
  host_accessor HostAccC(BufferC);

  for (size_t i = 0; i < Size; i++) {
    assert(check_value(i, ReferenceA[i], HostAccA[i], "HostAccA"));
    assert(check_value(i, ReferenceB[i], HostAccB[i], "HostAccB"));
    assert(check_value(i, ReferenceC[i], HostAccC[i], "HostAccC"));
  }

  return 0;
}
