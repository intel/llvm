// Tests creating a dotp operation which uses a sycl reduction with buffers.

#include "../graph_common.hpp"

int main() {
  queue Queue{};

  int DotpData = 0;

  const size_t N = 10;
  std::vector<int> XData(N);
  std::vector<int> YData(N);
  std::vector<int> ZData(N);

  buffer DotpBuf(&DotpData, range<1>(1));
  DotpBuf.set_write_back(false);

  buffer XBuf(XData);
  XBuf.set_write_back(false);
  buffer YBuf(YData);
  YBuf.set_write_back(false);
  buffer ZBuf(ZData);
  ZBuf.set_write_back(false);
  {
    exp_ext::command_graph Graph{
        Queue.get_context(),
        Queue.get_device(),
        {exp_ext::property::graph::assume_buffer_outlives_graph{}}};

    auto NodeI = add_node(Graph, Queue, [&](handler &CGH) {
      auto X = XBuf.get_access(CGH);
      auto Y = YBuf.get_access(CGH);
      auto Z = ZBuf.get_access(CGH);
      CGH.parallel_for(N, [=](id<1> it) {
        X[it] = 1;
        Y[it] = 2;
        Z[it] = 3;
      });
    });

    auto NodeA = add_node(
        Graph, Queue,
        [&](handler &CGH) {
          auto X = XBuf.get_access(CGH);
          auto Y = YBuf.get_access(CGH);
          CGH.parallel_for(range<1>{N}, [=](id<1> it) {
            X[it] = Alpha * X[it] + Beta * Y[it];
          });
        },
        NodeI);

    auto NodeB = add_node(
        Graph, Queue,
        [&](handler &CGH) {
          auto Y = YBuf.get_access(CGH);
          auto Z = ZBuf.get_access(CGH);
          CGH.parallel_for(range<1>{N}, [=](id<1> it) {
            Z[it] = Gamma * Z[it] + Beta * Y[it];
          });
        },
        NodeI);

    auto NodeC = add_node(
        Graph, Queue,
        [&](handler &CGH) {
          auto Dotp = DotpBuf.get_access(CGH);
          auto X = XBuf.get_access(CGH);
          auto Z = ZBuf.get_access(CGH);
          CGH.parallel_for(range<1>{N}, reduction(DotpBuf, CGH, 0, std::plus()),
                           [=](id<1> it, auto &Sum) { Sum += X[it] * Z[it]; });
        },
        NodeA, NodeB);

    auto ExecGraph = Graph.finalize();

    // Using shortcut for executing a graph of commands
    Queue.ext_oneapi_graph(ExecGraph).wait();
  }

  host_accessor HostAcc(DotpBuf);
  assert(HostAcc[0] == dotp_reference_result(N));

  return 0;
}
