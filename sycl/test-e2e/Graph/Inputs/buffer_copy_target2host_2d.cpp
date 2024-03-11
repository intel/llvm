// Tests adding 2d buffer copy -- Target to Host (rect read path) --  nodes
// and submitting the graph.

#include "../graph_common.hpp"

int main() {
  queue Queue{};

  using T = int;

  std::vector<T> DataA(Size * Size), DataB(Size * Size);
  std::iota(DataA.begin(), DataA.end(), 1);
  std::iota(DataB.begin(), DataB.end(), 1000);

  std::vector<T> ReferenceA(DataA), ReferenceB(DataB);
  for (size_t i = 0; i < Size * Size; i++) {
    ReferenceA[i] = DataA[i];
    ReferenceB[i] = DataA[i];
  }

  // Make the buffers 2D so we can test the rect read path
  buffer BufferA{DataA.data(), range<2>(Size, Size)};
  BufferA.set_write_back(false);

  {
    exp_ext::command_graph Graph{
        Queue.get_context(),
        Queue.get_device(),
        {exp_ext::property::graph::assume_buffer_outlives_graph{}}};

    auto NodeA = add_node(Graph, Queue, [&](handler &CGH) {
      auto AccA = BufferA.get_access<access::mode::read>(CGH);
      CGH.copy(AccA, DataB.data());
    });

    auto GraphExec = Graph.finalize();
    Queue.submit([&](handler &CGH) { CGH.ext_oneapi_graph(GraphExec); }).wait();
  }

  host_accessor HostAccA(BufferA);

  for (size_t i = 0; i < Size * Size; i++) {
    assert(check_value(i, ReferenceA[i], DataA[i], "DataA"));
    assert(check_value(i, ReferenceB[i], DataB[i], "DataB"));
  }

  return 0;
}
