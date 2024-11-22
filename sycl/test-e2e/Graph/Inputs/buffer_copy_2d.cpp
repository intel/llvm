// Tests adding buffer 2d copy nodes and submitting
// the graph.

#include "../graph_common.hpp"

#include <cmath>

int main() {
  queue Queue{};

  using T = int;

  const T ModValue = 7;

  const size_t SizeX = std::sqrt(Size);
  const size_t SizeY = SizeX;
  std::vector<T> DataA(Size), DataB(Size), DataC(Size);

  std::iota(DataA.begin(), DataA.end(), 1);
  std::iota(DataB.begin(), DataB.end(), 10);
  std::iota(DataC.begin(), DataC.end(), 1000);

  // Create reference data for output
  std::vector<T> ReferenceA(DataA), ReferenceB(DataB), ReferenceC(DataC);
  for (size_t i = 0; i < Iterations; i++) {
    for (size_t j = 0; j < Size; j++) {
      ReferenceA[j] = ReferenceB[j];
      ReferenceA[j] += ModValue;
      ReferenceB[j] = ReferenceA[j];
      ReferenceB[j] += (ModValue + 1);
      ReferenceC[j] = ReferenceB[j];
    }
  }

  // Make the buffers 2D so we can test the rect copy path
  buffer BufferA{DataA.data(), range<2>(SizeX, SizeY)};
  BufferA.set_write_back(false);
  buffer BufferB{DataB.data(), range<2>(SizeX, SizeY)};
  BufferB.set_write_back(false);
  buffer BufferC{DataC.data(), range<2>(SizeX, SizeY)};
  BufferC.set_write_back(false);
  {
    exp_ext::command_graph Graph{
        Queue.get_context(),
        Queue.get_device(),
        {exp_ext::property::graph::assume_buffer_outlives_graph{}}};

    // Copy from B to A
    auto NodeA = add_node(Graph, Queue, [&](handler &CGH) {
      auto AccA = BufferA.get_access(CGH);
      auto AccB = BufferB.get_access(CGH);
      CGH.copy(AccB, AccA);
    });

    // Read & write A
    auto NodeB = add_node(
        Graph, Queue,
        [&](handler &CGH) {
          auto AccA = BufferA.get_access(CGH);
          CGH.parallel_for(range<2>(SizeX, SizeY),
                           [=](item<2> id) { AccA[id] += ModValue; });
        },
        NodeA);

    // Read & write B
    auto NodeModB = add_node(
        Graph, Queue,
        [&](handler &CGH) {
          auto AccB = BufferB.get_access(CGH);
          CGH.parallel_for(range<2>(SizeX, SizeY),
                           [=](item<2> id) { AccB[id] += ModValue; });
        },
        NodeA);

    // memcpy from A to B
    auto NodeC = add_node(
        Graph, Queue,
        [&](handler &CGH) {
          auto AccA = BufferA.get_access(CGH);
          auto AccB = BufferB.get_access(CGH);
          CGH.copy(AccA, AccB);
        },
        NodeModB);

    // Read and write B
    auto NodeD = add_node(
        Graph, Queue,
        [&](handler &CGH) {
          auto AccB = BufferB.get_access(CGH);
          CGH.parallel_for(range<2>(SizeX, SizeY),
                           [=](item<2> id) { AccB[id] += (ModValue + 1); });
        },
        NodeC);

    // Copy from B to C
    add_node(
        Graph, Queue,
        [&](handler &CGH) {
          auto AccB = BufferB.get_access(CGH);
          auto AccC = BufferC.get_access(CGH);
          CGH.copy(AccB, AccC);
        },
        NodeD);

    auto GraphExec = Graph.finalize();

    for (unsigned n = 0; n < Iterations; n++) {
      Queue.submit([&](handler &CGH) { CGH.ext_oneapi_graph(GraphExec); });
    }
    Queue.wait_and_throw();
  }

  host_accessor HostAccA(BufferA);
  host_accessor HostAccB(BufferB);
  host_accessor HostAccC(BufferC);

  for (size_t i = 0; i < SizeX; i++) {
    for (size_t j = 0; j < SizeY; j++) {
      const size_t index = i * SizeY + j;
      assert(check_value(index, ReferenceA[index], HostAccA[i][j], "HostAccA"));
      assert(check_value(index, ReferenceB[index], HostAccB[i][j], "HostAccB"));
      assert(check_value(index, ReferenceC[index], HostAccC[i][j], "HostAccC"));
    }
  }

  return 0;
}
