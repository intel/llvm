// Tests enqueueing an eager buffer submission followed by two executions
// of the same graph which also use the buffer.

#include "../graph_common.hpp"

int main() {
  queue Queue{};

  const size_t N = 10;
  std::vector<int> Arr(N, 0);

  buffer<int> Buf{N};
  Buf.set_write_back(false);

  {
    // Buffer elements set to 8
    Queue.submit([&](handler &CGH) {
      auto Acc = Buf.get_access(CGH);
      CGH.parallel_for(range<1>{N}, [=](id<1> idx) {
        size_t i = idx;
        Acc[i] = 8;
      });
    });

    // Create graph than adds 2 to buffer elements
    exp_ext::command_graph Graph{
        Queue.get_context(),
        Queue.get_device(),
        {exp_ext::property::graph::assume_buffer_outlives_graph{}}};

    add_node(Graph, Queue, [&](handler &CGH) {
      auto Acc = Buf.get_access(CGH);
      CGH.parallel_for(range<1>{N}, [=](id<1> idx) {
        size_t i = idx;
        Acc[i] += 2;
      });
    });

    auto ExecGraph = Graph.finalize();

    // Buffer elements set to 10
    Queue.ext_oneapi_graph(ExecGraph);

    // Buffer elements set to 12
    Queue.ext_oneapi_graph(ExecGraph);

    // Copy results back
    Queue.submit([&](handler &CGH) {
      auto Acc = Buf.get_access(CGH);
      CGH.copy(Acc, Arr.data());
    });
    Queue.wait();
  }

  const int Expected = 12;
  for (size_t i = 0; i < N; i++) {
    assert(check_value(i, Expected, Arr[i], "Arr"));
  }

  return 0;
}
