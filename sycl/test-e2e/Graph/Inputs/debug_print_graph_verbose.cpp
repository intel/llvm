// Tests adding nodes to a graph and submitting the graph
// using buffers accessors for inputs and outputs.

#include "../graph_common.hpp"

int main() {
  queue Queue{{sycl::ext::intel::property::queue::no_immediate_command_list{}}};

  if (!are_graphs_supported(Queue)) {
    return 0;
  }

  using T = unsigned short;

  std::vector<T> DataA(Size), DataB(Size), DataC(Size), DataD(Size);

  std::iota(DataA.begin(), DataA.end(), 1);
  std::iota(DataB.begin(), DataB.end(), 10);
  std::iota(DataC.begin(), DataC.end(), 1000);
  std::iota(DataD.begin(), DataD.end(), 1);

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

    // Copy from B to A
    add_node(Graph, Queue, [&](handler &CGH) {
      auto AccA = BufferA.get_access(CGH);
      auto AccB = BufferB.get_access(CGH);
      CGH.copy(AccB, AccA);
    });

    auto Last = add_node(Graph, Queue, [&](handler &CGH) {
      auto AccA = BufferC.get_access<access::mode::read>(CGH);
      CGH.copy(AccA, DataD.data());
    });

    add_empty_node(Graph, Queue, Last);

    Graph.print_graph("graph_verbose.dot", true);
  }

  return 0;
}
