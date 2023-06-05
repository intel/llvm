// REQUIRES: level_zero, gpu
// RUN: %{build} -o %t.out
// RUN: %{run} %t.out

// XFAIL: *
// Subgraph doesn't work properly in second parent graph

// Tests adding an executable graph object as a sub-graph of two different
// parent graphs.

#include "../graph_common.hpp"

int main() {
  queue Queue{gpu_selector_v};

  exp_ext::command_graph GraphA{Queue.get_context(), Queue.get_device()};
  exp_ext::command_graph GraphB{Queue.get_context(), Queue.get_device()};
  exp_ext::command_graph SubGraph{Queue.get_context(), Queue.get_device()};

  const size_t N = 10;
  float *X = malloc_device<float>(N, Queue);

  SubGraph.begin_recording(Queue);

  auto S1 = Queue.submit([&](handler &CGH) {
    CGH.parallel_for(N, [=](id<1> it) {
      const size_t i = it[0];
      X[i] *= 2.0f;
    });
  });

  Queue.submit([&](handler &CGH) {
    CGH.depends_on(S1);
    CGH.parallel_for(N, [=](id<1> it) {
      const size_t i = it[0];
      X[i] += 0.5f;
    });
  });

  SubGraph.end_recording(Queue);

  auto ExecSubGraph = SubGraph.finalize();

  GraphA.begin_recording(Queue);

  auto A1 = Queue.submit([&](handler &CGH) {
    CGH.parallel_for(N, [=](id<1> it) {
      const size_t i = it[0];
      X[i] = 1.0f;
    });
  });

  auto A2 = Queue.submit([&](handler &CGH) {
    CGH.depends_on(A1);
    CGH.ext_oneapi_graph(ExecSubGraph);
  });

  Queue.submit([&](handler &CGH) {
    CGH.depends_on(A2);
    CGH.parallel_for(range<1>{N}, [=](id<1> it) {
      const size_t i = it[0];
      X[i] *= -1.0f;
    });
  });

  GraphA.end_recording();

  auto ExecGraphA = GraphA.finalize();

  GraphB.begin_recording(Queue);

  auto B1 = Queue.submit([&](handler &CGH) {
    CGH.parallel_for(N, [=](id<1> it) {
      const size_t i = it[0];
      X[i] = static_cast<float>(i);
    });
  });

  auto B2 = Queue.submit([&](handler &CGH) {
    CGH.depends_on(B1);
    CGH.ext_oneapi_graph(ExecSubGraph);
  });

  Queue.submit([&](handler &CGH) {
    CGH.depends_on(B2);
    CGH.parallel_for(range<1>{N}, [=](id<1> it) {
      const size_t i = it[0];
      X[i] *= X[i];
    });
  });

  GraphB.end_recording();
  auto ExecGraphB = GraphB.finalize();

  auto EventA1 =
      Queue.submit([&](handler &CGH) { CGH.ext_oneapi_graph(ExecGraphA); });
  std::vector<float> OutputA(N);
  auto EventA2 = Queue.memcpy(OutputA.data(), X, N * sizeof(float), EventA1);

  auto EventB1 = Queue.submit([&](handler &CGH) {
    CGH.depends_on(EventA2);
    CGH.ext_oneapi_graph(ExecGraphB);
  });
  std::vector<float> OutputB(N);
  Queue.memcpy(OutputB.data(), X, N * sizeof(float), EventB1);
  Queue.wait();

  auto refB = [](size_t i) {
    float result = static_cast<float>(i);
    result = result * 2.0f + 0.5f;
    result *= result;
    return result;
  };

  for (size_t i = 0; i < N; i++) {
    assert(OutputA[i] == -2.5f);
    assert(OutputB[i] == refB(i));
  }

  sycl::free(X, Queue);

  return 0;
}
