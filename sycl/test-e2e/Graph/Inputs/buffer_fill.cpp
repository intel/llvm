// Tests adding a Buffer fill operation as a graph node.

#include "../graph_common.hpp"

int main() {

  queue Queue;
  const size_t N = 10;
  const float Pattern = 3.14f;
  std::vector<float> Data(N);
  buffer<float> Buffer(Data);
  Buffer.set_write_back(false);
  {
    exp_ext::command_graph Graph{
        Queue.get_context(),
        Queue.get_device(),
        {exp_ext::property::graph::assume_buffer_outlives_graph{},
         exp_ext::property::graph::assume_data_outlives_buffer{}}};

    auto NodeA = add_node(Graph, Queue, [&](handler &CGH) {
      auto Acc = Buffer.get_access(CGH);
      CGH.fill(Acc, Pattern);
    });

    auto ExecGraph = Graph.finalize();

    Queue.submit([&](handler &CGH) { CGH.ext_oneapi_graph(ExecGraph); }).wait();
  }
  host_accessor HostData(Buffer);
  for (int i = 0; i < N; i++)
    assert(HostData[i] == Pattern);

  return 0;
}
