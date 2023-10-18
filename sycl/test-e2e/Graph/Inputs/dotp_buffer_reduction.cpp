// Tests creating a dotp operation which uses a sycl reduction with buffers.

#include "../graph_common.hpp"

int main() {

  queue Queue;

  exp_ext::command_graph Graph{Queue.get_context(), Queue.get_device()};

  float DotpData = 0.f;

  const size_t N = 10;
  std::vector<float> XData(N);
  std::vector<float> YData(N);
  std::vector<float> ZData(N);

  {
    buffer DotpBuf(&DotpData, range<1>(1));
    DotpBuf.set_write_back(false);

    buffer XBuf(XData);
    XBuf.set_write_back(false);
    buffer YBuf(YData);
    YBuf.set_write_back(false);
    buffer ZBuf(ZData);
    ZBuf.set_write_back(false);

    auto NodeI = add_node(Graph, Queue, [&](handler &CGH) {
      auto X = XBuf.get_access(CGH);
      auto Y = YBuf.get_access(CGH);
      auto Z = ZBuf.get_access(CGH);
      CGH.parallel_for(N, [=](id<1> it) {
        X[it] = 1.0f;
        Y[it] = 2.0f;
        Z[it] = 3.0f;
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
          CGH.parallel_for(range<1>{N},
                           reduction(DotpBuf, CGH, 0.0f, std::plus()),
                           [=](id<1> it, auto &Sum) { Sum += X[it] * Z[it]; });
        },
        NodeA, NodeB);

    auto ExecGraph = Graph.finalize();

    // Using shortcut for executing a graph of commands
    Queue.ext_oneapi_graph(ExecGraph).wait();

    host_accessor HostAcc(DotpBuf);
    assert(HostAcc[0] == dotp_reference_result(N));
  }

  return 0;
}
