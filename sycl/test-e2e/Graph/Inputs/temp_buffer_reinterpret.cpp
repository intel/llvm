// This test creates a temporary buffer (which is reinterpreted from the main
// application buffers) which is used in kernels but destroyed before
// finalization and execution of the graph. The original buffers lifetime
// extends until after execution of the graph.

#include "../graph_common.hpp"

int main() {
  queue Queue{};

  using T = int;

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

    {
      // Create some temporary buffers only for adding nodes
      auto BufferA2 = BufferA.reinterpret<T, 1>(BufferA.get_range());
      auto BufferB2 = BufferB.reinterpret<T, 1>(BufferB.get_range());
      auto BufferC2 = BufferC.reinterpret<T, 1>(BufferC.get_range());

      add_nodes(Graph, Queue, Size, BufferA2, BufferB2, BufferC2);
    }
    auto GraphExec = Graph.finalize();

    for (unsigned n = 0; n < Iterations; n++) {
      Queue.submit([&](handler &CGH) { CGH.ext_oneapi_graph(GraphExec); });
    }

    Queue.copy(BufferA.get_access(), DataA.data());
    Queue.copy(BufferB.get_access(), DataB.data());
    Queue.copy(BufferC.get_access(), DataC.data());
    // Perform a wait on all graph submissions.
    Queue.wait_and_throw();
  }

  for (size_t i = 0; i < Size; i++) {
    assert(check_value(i, ReferenceA[i], DataA[i], "DataA"));
    assert(check_value(i, ReferenceB[i], DataB[i], "DataB"));
    assert(check_value(i, ReferenceC[i], DataC[i], "DataC"));
  }

  return 0;
}
