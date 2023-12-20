// Tests adding nodes to a graph and submitting the graph
// using buffers accessors for inputs and outputs.

#include "../graph_common.hpp"

int main() {
  queue Queue{{sycl::ext::intel::property::queue::no_immediate_command_list{}}};

  if (!are_graphs_supported(Queue)) {
    return 0;
  }

  using T = unsigned short;

  std::vector<T> DataA(Size), DataB(Size), DataC(Size);
  std::vector<T> DataA2D(Size * Size), DataB2D(Size * Size);

  std::iota(DataA.begin(), DataA.end(), 1);
  std::iota(DataB.begin(), DataB.end(), 10);
  std::iota(DataC.begin(), DataC.end(), 1000);
  std::iota(DataA2D.begin(), DataA2D.end(), 1);
  std::iota(DataB2D.begin(), DataB2D.end(), 10);

  buffer<T> BufferA{DataA.data(), range<1>{DataA.size()}};
  BufferA.set_write_back(false);
  buffer<T> BufferB{DataB.data(), range<1>{DataB.size()}};
  BufferB.set_write_back(false);
  buffer<T> BufferC{DataC.data(), range<1>{DataC.size()}};
  BufferC.set_write_back(false);
  buffer BufferA2D{DataA2D.data(), range<2>(Size, Size)};
  BufferA2D.set_write_back(false);
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

    add_node(Graph, Queue, [&](handler &CGH) {
      auto AccA = BufferA2D.get_access<access::mode::read>(CGH);
      CGH.copy(AccA, DataB2D.data());
    });

    add_node(Graph, Queue, [&](handler &CGH) { /* empty node */ });

    Graph.print_graph("graph_verbose.dot", true);
  }

  return 0;
}
