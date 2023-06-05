// REQUIRES: level_zero, gpu
// RUN: %{build} -o %t.out
// RUN: %{run} %t.out

// XFAIL: *
// Subgraph nodes get parent graph precessor nodes when added as a subgraph
// which affects stand alone execution.

// Tests creating a parent graph with the same sub-graph interleaved with
// other nodes.

#include "../graph_common.hpp"

int main() {
  queue Queue{gpu_selector_v};

  exp_ext::command_graph Graph{Queue.get_context(), Queue.get_device()};
  exp_ext::command_graph SubGraph{Queue.get_context(), Queue.get_device()};

  const size_t N = 10;
  float *X = malloc_device<float>(N, Queue);

  auto S1 = SubGraph.add([&](handler &CGH) {
    CGH.parallel_for(N, [=](id<1> it) {
      const size_t i = it[0];
      X[i] *= 2.0f;
    });
  });

  SubGraph.add(
      [&](handler &CGH) {
        CGH.parallel_for(N, [=](id<1> it) {
          const size_t i = it[0];
          X[i] += 0.5f;
        });
      },
      {exp_ext::property::node::depends_on(S1)});

  auto ExecSubGraph = SubGraph.finalize();

  auto G1 = Graph.add([&](handler &CGH) {
    CGH.parallel_for(N, [=](id<1> it) {
      const size_t i = it[0];
      X[i] = 1.0f;
    });
  });

  auto G2 = Graph.add([&](handler &CGH) { CGH.ext_oneapi_graph(ExecSubGraph); },
                      {exp_ext::property::node::depends_on(G1)});

  Graph.add(
      [&](handler &CGH) {
        CGH.parallel_for(range<1>{N}, [=](id<1> it) {
          const size_t i = it[0];
          X[i] *= -1.0f;
        });
      },
      {exp_ext::property::node::depends_on(G2)});

  auto ExecGraph = Graph.finalize();

  auto Event1 = Queue.submit([&](handler &CGH) {
    CGH.parallel_for(N, [=](id<1> it) {
      const size_t i = it[0];
      X[i] = 3.14f;
    });
  });

  auto Event2 = Queue.submit([&](handler &CGH) {
    CGH.depends_on(Event1);
    CGH.ext_oneapi_graph(ExecGraph);
  });

  std::vector<float> Output(N);
  Queue.memcpy(Output.data(), X, N * sizeof(float), Event2).wait();

  const float ref = 3.14f * 2.0f + 0.5f;
  for (size_t i = 0; i < N; i++) {
    assert(Output[i] == ref);
  }

  sycl::free(X, Queue);

  return 0;
}
