// Tests adding a buffer copy node and submitting the graph.

#include "../graph_common.hpp"

int main() {
  queue Queue{{sycl::ext::intel::property::queue::no_immediate_command_list{}}};

  if (!are_graphs_supported(Queue)) {
    return 0;
  }

  using T = int;

  const T ModValue = 7;
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
      ReferenceB[j] += ModValue;
      ReferenceC[j] = ReferenceB[j];
    }
  }

  buffer BufferA{DataA};
  BufferA.set_write_back(false);
  buffer BufferB{DataB};
  BufferB.set_write_back(false);
  buffer BufferC{DataC};
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
          CGH.parallel_for(range<1>(Size), [=](item<1> id) {
            auto LinID = id.get_linear_id();
            AccA[LinID] += ModValue;
          });
        },
        NodeA);

    // Read & write B
    auto NodeModB = add_node(
        Graph, Queue,
        [&](handler &CGH) {
          auto AccB = BufferB.get_access(CGH);
          CGH.parallel_for(range<1>(Size), [=](item<1> id) {
            auto LinID = id.get_linear_id();
            AccB[LinID] += ModValue;
          });
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
        NodeB, NodeModB);

    // Read and write B
    auto NodeD = add_node(
        Graph, Queue,
        [&](handler &CGH) {
          auto AccB = BufferB.get_access(CGH);
          CGH.parallel_for(range<1>(Size), [=](item<1> id) {
            auto LinID = id.get_linear_id();
            AccB[LinID] += ModValue;
          });
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

    event Event;
    for (unsigned n = 0; n < Iterations; n++) {
      Event =
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
