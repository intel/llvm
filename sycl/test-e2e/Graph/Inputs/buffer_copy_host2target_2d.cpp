// Tests adding 2d buffer copy -- Host to Target (write path) --  nodes
// and submitting the graph.

#include "../graph_common.hpp"

#include <cmath>

int main() {
  queue Queue{};

  using T = int;

  const size_t SizeX = std::sqrt(Size);
  const size_t SizeY = SizeX;
  std::vector<T> DataA(Size), DataB(Size);
  std::iota(DataA.begin(), DataA.end(), 1);
  std::iota(DataB.begin(), DataB.end(), 1000);

  std::vector<T> ReferenceA(DataA);
  for (size_t i = 0; i < Size; i++) {
    ReferenceA[i] = DataB[i];
  }

  // Make the buffers 2D so we can test the rect write path
  buffer BufferA{DataA.data(), range<2>(SizeX, SizeY)};
  BufferA.set_write_back(false);

  {
    exp_ext::command_graph Graph{
        Queue.get_context(),
        Queue.get_device(),
        {exp_ext::property::graph::assume_buffer_outlives_graph{}}};

    auto NodeA = add_node(Graph, Queue, [&](handler &CGH) {
      auto AccA = BufferA.get_access<access::mode::write>(CGH);
      CGH.copy(DataB.data(), AccA);
    });

    auto GraphExec = Graph.finalize();
    Queue.submit([&](handler &CGH) { CGH.ext_oneapi_graph(GraphExec); }).wait();
  }
  host_accessor HostAccA(BufferA);

  for (size_t i = 0; i < SizeX; i++) {
    for (size_t j = 0; j < SizeY; j++) {
      const size_t index = i * SizeY + j;
      assert(check_value(index, ReferenceA[index], HostAccA[i][j], "HostAccA"));
    }
  }

  return 0;
}
