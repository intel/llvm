// Tests adding buffer copy offset -- Host to Target (write path) --  nodes
// and submitting the graph.

#include "../graph_common.hpp"

int main() {
  queue Queue{{sycl::ext::intel::property::queue::no_immediate_command_list{}}};

  using T = int;

  std::vector<T> DataA(Size + Offset), DataB(Size);
  std::iota(DataA.begin(), DataA.end(), 1);
  std::iota(DataB.begin(), DataB.end(), 1000);

  std::vector<T> ReferenceA(Size + Offset);
  for (size_t i = 0; i < Size + Offset; i++) {
    if (i < Offset)
      ReferenceA[i] = DataA[i];
    else
      ReferenceA[i] = DataB[i - Offset];
  }

  buffer<T, 1> BufferA(DataA.data(), range<1>(Size + Offset));
  BufferA.set_write_back(false);

  {
    exp_ext::command_graph Graph{
        Queue.get_context(),
        Queue.get_device(),
        {exp_ext::property::graph::assume_buffer_outlives_graph{}}};

    auto NodeA = add_node(Graph, Queue, [&](handler &CGH) {
      auto AccA = BufferA.get_access<access::mode::write>(CGH, range<1>(Size),
                                                          id<1>(Offset));
      CGH.copy(DataB.data(), AccA);
    });

    auto GraphExec = Graph.finalize();
    Queue.submit([&](handler &CGH) { CGH.ext_oneapi_graph(GraphExec); }).wait();
  }

  host_accessor HostAccA(BufferA);

  for (size_t i = 0; i < Size + Offset; i++) {
    assert(ReferenceA[i] == HostAccA[i]);
  }

  return 0;
}
