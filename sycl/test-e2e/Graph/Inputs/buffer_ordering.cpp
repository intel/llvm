// Tests that buffer accessors exhibit the correct behaviour when:
// * A node is added to the graph between two queue submissions which
//   use the same buffer, but are not added to the graph.
//
// * A queue submission using the same buffer is made after finalization
//   of the graph, but before graph execution.
//
// * The graph is submitted for execution twice separated by a queue
//   submission using the same buffer, this should respect dependencies and
//   create the correct ordering.

#include "../graph_common.hpp"

int main() {
  queue Queue{};

  if (!are_graphs_supported(Queue)) {
    return 0;
  }

  const size_t N = 10;
  std::vector<int> Arr(N, 0);

  buffer<int> Buf{N};
  Buf.set_write_back(false);

  {
    exp_ext::command_graph Graph{
        Queue.get_context(),
        Queue.get_device(),
        {exp_ext::property::graph::assume_buffer_outlives_graph{}}};

    // Buffer elements set to 3
    Queue.submit([&](handler &CGH) {
      auto Acc = Buf.get_access(CGH);
      CGH.parallel_for(range<1>{N}, [=](id<1> idx) {
        size_t i = idx;
        Acc[i] = 3;
      });
    });

    add_node(Graph, Queue, [&](handler &CGH) {
      auto Acc = Buf.get_access(CGH);
      CGH.parallel_for(range<1>{N}, [=](id<1> idx) {
        size_t i = idx;
        Acc[i] += 2;
      });
    });

    const int Zero = 0;
    for (size_t i = 0; i < N; i++) {
      assert(check_value(i, Zero, Arr[i], "Arr"));
    }

    // Buffer elements set to 4
    Queue.submit([&](handler &CGH) {
      auto Acc = Buf.get_access(CGH);
      CGH.parallel_for(range<1>{N}, [=](id<1> idx) {
        size_t i = idx;
        Acc[i] += 1;
      });
    });

    auto ExecGraph = Graph.finalize();

    for (size_t i = 0; i < N; i++) {
      assert(check_value(i, Zero, Arr[i], "Arr"));
    }

    // Buffer elements set to 8
    Queue.submit([&](handler &CGH) {
      auto Acc = Buf.get_access(CGH);
      CGH.parallel_for(range<1>{N}, [=](id<1> idx) {
        size_t i = idx;
        Acc[i] *= 2;
      });
    });

    // Buffer elements set to 10
    auto Event =
        Queue.submit([&](handler &CGH) { CGH.ext_oneapi_graph(ExecGraph); });

    // Buffer elements set to 20
    Queue.submit([&](handler &CGH) {
      auto Acc = Buf.get_access(CGH);
      CGH.parallel_for(range<1>{N}, [=](id<1> idx) {
        size_t i = idx;
        Acc[i] *= 2;
      });
    });

    // Buffer elements set to 22
    Queue.submit([&](handler &CGH) { CGH.ext_oneapi_graph(ExecGraph); });

    Queue.submit([&](handler &CGH) {
      auto Acc = Buf.get_access(CGH);
      CGH.copy(Acc, Arr.data());
    });
    Queue.wait();
  }

  const int Expected = 22;
  for (size_t i = 0; i < N; i++) {
    assert(check_value(i, Expected, Arr[i], "Arr"));
  }

  return 0;
}
