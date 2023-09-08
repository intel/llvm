// REQUIRES: cuda || level_zero, gpu
// RUN: %{build} -o %t.out
// RUN: %{run} %t.out
// RUN: %if ext_oneapi_level_zero %{env ZE_DEBUG=4 %{run} %t.out 2>&1 | FileCheck %s %}
//
// CHECK-NOT: LEAK

// Tests capturing a dotp operation through queue recording using buffers.
// The second run is to check that there are no leaks reported with the embedded
// ZE_DEBUG=4 testing capability.

#include "../graph_common.hpp"

int main() {
  float DotpData = 0.f;

  const size_t N = 10;
  std::vector<float> XData(N);
  std::vector<float> YData(N);
  std::vector<float> ZData(N);

  buffer DotpBuf(&DotpData, range<1>(1));
  DotpBuf.set_write_back(false);

  buffer XBuf(XData);
  XBuf.set_write_back(false);
  buffer YBuf(YData);
  YBuf.set_write_back(false);
  buffer ZBuf(ZData);
  ZBuf.set_write_back(false);

  {
    queue Queue{gpu_selector_v};

    exp_ext::command_graph Graph{
        Queue.get_context(),
        Queue.get_device(),
        {exp_ext::property::graph::assume_buffer_outlives_graph{}}};

    Graph.begin_recording(Queue);

    Queue.submit([&](handler &CGH) {
      auto X = XBuf.get_access(CGH);
      auto Y = YBuf.get_access(CGH);
      auto Z = ZBuf.get_access(CGH);
      CGH.parallel_for(N, [=](id<1> it) {
        const size_t i = it[0];
        X[i] = 1.0f;
        Y[i] = 2.0f;
        Z[i] = 3.0f;
      });
    });

    Queue.submit([&](handler &CGH) {
      auto X = XBuf.get_access(CGH);
      auto Y = YBuf.get_access(CGH);
      CGH.parallel_for(range<1>{N}, [=](id<1> it) {
        const size_t i = it[0];
        X[i] = Alpha * X[i] + Beta * Y[i];
      });
    });

    Queue.submit([&](handler &CGH) {
      auto Y = YBuf.get_access(CGH);
      auto Z = ZBuf.get_access(CGH);
      CGH.parallel_for(range<1>{N}, [=](id<1> it) {
        const size_t i = it[0];
        Z[i] = Gamma * Z[i] + Beta * Y[i];
      });
    });

    Queue.submit([&](handler &CGH) {
      auto Dotp = DotpBuf.get_access(CGH);
      auto X = XBuf.get_access(CGH);
      auto Z = ZBuf.get_access(CGH);
      CGH.single_task([=]() {
        for (size_t j = 0; j < N; j++) {
          Dotp[0] += X[j] * Z[j];
        }
      });
    });
    Graph.end_recording();

    auto ExecGraph = Graph.finalize();

    Queue.submit([&](handler &CGH) { CGH.ext_oneapi_graph(ExecGraph); }).wait();
  }

  host_accessor HostAcc(DotpBuf);
  assert(HostAcc[0] == dotp_reference_result(N));

  return 0;
}
