// REQUIRES: level_zero, gpu
// RUN: %{build} -o %t.out
// RUN: %{run} %t.out
// RUN: %if ext_oneapi_level_zero %{env ZE_DEBUG=4 %{run} %t.out 2>&1 | FileCheck %s %}
//
// CHECK-NOT: LEAK

// Expected fail as reduction support is not complete.
// XFAIL: *

// Modified version of the dotp example which submits which contains a sycl
// reduction as well as a sub-graph.
// The second run is to check that there are no leaks reported with the embedded
// ZE_DEBUG=4 testing capability.

#include "../graph_common.hpp"

int main() {
  queue Queue{gpu_selector_v};

  exp_ext::command_graph Graph{Queue.get_context(), Queue.get_device()};
  exp_ext::command_graph SubGraph{Queue.get_context(), Queue.get_device()};

  float *Dotp = malloc_device<float>(1, Queue);

  const size_t N = 10;
  float *X = malloc_device<float>(N, Queue);
  float *Y = malloc_device<float>(N, Queue);
  float *Z = malloc_device<float>(N, Queue);

  auto NodeI = Graph.add([&](handler &CGH) {
    CGH.parallel_for(N, [=](id<1> it) {
      const size_t i = it[0];
      X[i] = 1.0f;
      Y[i] = 2.0f;
      Z[i] = 3.0f;
    });
  });

  auto NodeA = SubGraph.add([&](handler &CGH) {
    CGH.parallel_for(range<1>{N}, [=](id<1> it) {
      const size_t i = it[0];
      X[i] = Alpha * X[i] + Beta * Y[i];
    });
  });

  auto NodeB = SubGraph.add([&](handler &CGH) {
    CGH.parallel_for(range<1>{N}, [=](id<1> it) {
      const size_t i = it[0];
      Z[i] = Gamma * Z[i] + Beta * Y[i];
    });
  });

  auto SubGraphExec = SubGraph.finalize();

  auto NodeSub =
      Graph.add([&](handler &CGH) { CGH.ext_oneapi_graph(SubGraphExec); },
                {exp_ext::property::node::depends_on(NodeI)});

  auto NodeC = Graph.add(
      [&](handler &CGH) {
        CGH.parallel_for(range<1>{N}, reduction(Dotp, 0.0f, std::plus()),
                         [=](id<1> it, auto &Sum) {
                           const size_t i = it[0];
                           Sum += X[i] * Z[i];
                         });
      },
      {exp_ext::property::node::depends_on(NodeSub)});

  auto ExecGraph = Graph.finalize();

  // Using shortcut for executing a graph of commands
  Queue.ext_oneapi_graph(ExecGraph).wait();

  float Output;
  Queue.memcpy(&Output, Dotp, sizeof(float)).wait();
  assert(Output == dotp_reference_result(N));

  sycl::free(Dotp, Queue);
  sycl::free(X, Queue);
  sycl::free(Y, Queue);
  sycl::free(Z, Queue);

  return 0;
}
