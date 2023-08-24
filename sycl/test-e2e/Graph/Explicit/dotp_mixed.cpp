// REQUIRES: cuda || level_zero, gpu
// RUN: %{build} -o %t.out
// RUN: %{run} %t.out
// RUN: %if ext_oneapi_level_zero %{env ZE_DEBUG=4 %{run} %t.out 2>&1 | FileCheck %s %}
//
// CHECK-NOT: LEAK

// Tests mixing buffers and USM in the same graph.
// The second run is to check that there are no leaks reported with the embedded
// ZE_DEBUG=4 testing capability.

#include "../graph_common.hpp"

int main() {
  queue Queue{gpu_selector_v};

  float DotpData = 0.f;

  const size_t N = 10;
  std::vector<float> XData(N);

  buffer DotpBuf(&DotpData, range<1>(1));
  DotpBuf.set_write_back(false);

  buffer XBuf(XData);
  XBuf.set_write_back(false);

  float *Y = malloc_device<float>(N, Queue);
  float *Z = malloc_device<float>(N, Queue);

  {
    exp_ext::command_graph Graph{
        Queue.get_context(),
        Queue.get_device(),
        {exp_ext::property::graph::assume_buffer_outlives_graph{},
         exp_ext::property::graph::assume_data_outlives_buffer{}}};

    auto NodeI = Graph.add([&](handler &CGH) {
      auto X = XBuf.get_access(CGH);
      CGH.parallel_for(N, [=](id<1> it) {
        const size_t i = it[0];
        X[i] = 1.0f;
        Y[i] = 2.0f;
        Z[i] = 3.0f;
      });
    });

    // Edge to NodeI from buffer accessor
    auto NodeA = Graph.add([&](handler &CGH) {
      auto X = XBuf.get_access(CGH);
      CGH.parallel_for(range<1>{N}, [=](id<1> it) {
        const size_t i = it[0];
        X[i] = Alpha * X[i] + Beta * Y[i];
      });
    });

    // Edge to NodeI explicitly added
    auto NodeB = Graph.add(
        [&](handler &CGH) {
          CGH.parallel_for(range<1>{N}, [=](id<1> it) {
            const size_t i = it[0];
            Z[i] = Gamma * Z[i] + Beta * Y[i];
          });
        },
        {exp_ext::property::node::depends_on(NodeI)});

    // Edge node_a from buffer accessor, and edge to NodeB explicitly added
    auto NodeC = Graph.add(
        [&](handler &CGH) {
          auto Dotp = DotpBuf.get_access(CGH);
          auto X = XBuf.get_access(CGH);
          CGH.single_task([=]() {
            for (size_t j = 0; j < N; j++) {
              Dotp[0] += X[j] * Z[j];
            }
          });
        },
        {exp_ext::property::node::depends_on(NodeB)});

    auto ExecGraph = Graph.finalize();

    // Using shortcut for executing a graph of commands
    Queue.ext_oneapi_graph(ExecGraph).wait();

    sycl::free(Y, Queue);
    sycl::free(Z, Queue);
  }

  host_accessor HostAcc(DotpBuf);
  assert(HostAcc[0] == dotp_reference_result(N));
  return 0;
}
