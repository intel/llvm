// Tests adding buffer copy nodes with offsets
// and submitting the graph.

#include "../graph_common.hpp"

int main() {
  queue Queue{};

  if (!are_graphs_supported(Queue)) {
    return 0;
  }

  using T = int;

  size_t OffsetSrc = 2 * size_t(Size / 4);
  size_t OffsetDst = size_t(Size / 4);
  std::vector<T> DataA(Size), DataB(Size);

  std::iota(DataA.begin(), DataA.end(), 1);
  std::iota(DataB.begin(), DataB.end(), 10000);

  // Create reference data for output
  std::vector<T> ReferenceA(DataA), ReferenceB(DataB);
  for (size_t j = 0; j < Size; j++) {
    ReferenceA[j] = DataA[j];
    ReferenceB[j] = DataB[j];
  }
  for (size_t j = OffsetDst; j < Size - (OffsetSrc - OffsetDst); j++) {
    ReferenceB[j] = DataA[(j - OffsetDst) + OffsetSrc];
  }

  buffer BufferA{DataA};
  BufferA.set_write_back(false);
  buffer BufferB{DataB};
  BufferB.set_write_back(false);

  {
    exp_ext::command_graph Graph{
        Queue.get_context(),
        Queue.get_device(),
        {exp_ext::property::graph::assume_buffer_outlives_graph{}}};

    // Copy from A to B
    auto NodeA = add_node(Graph, Queue, [&](handler &CGH) {
      auto AccA = BufferA.get_access<access::mode::read_write>(
          CGH, range<1>(Size - OffsetSrc), id<1>(OffsetSrc));
      auto AccB = BufferB.get_access<access::mode::read_write>(
          CGH, range<1>(Size - OffsetDst), id<1>(OffsetDst));
      CGH.copy(AccA, AccB);
    });

    auto GraphExec = Graph.finalize();
    Queue.submit([&](handler &CGH) { CGH.ext_oneapi_graph(GraphExec); }).wait();
  }

  host_accessor HostAccA(BufferA);
  host_accessor HostAccB(BufferB);

  for (size_t i = 0; i < Size; i++) {
    assert(check_value(i, ReferenceA[i], HostAccA[i], "HostAccA"));
    assert(check_value(i, ReferenceB[i], HostAccB[i], "HostAccB"));
  }

  return 0;
}
