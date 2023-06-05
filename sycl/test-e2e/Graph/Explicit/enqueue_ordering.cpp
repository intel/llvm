// REQUIRES: level_zero, gpu
// RUN: %{build} -o %t.out
// RUN: %{run} %t.out

// Test submitting the same graph twice with another command in between, this
// intermediate command depends on the first submission of the graph, and
// is a dependency of the second submission of the graph.

#include "../graph_common.hpp"
int main() {

  queue Queue{gpu_selector_v};

  exp_ext::command_graph Graph{Queue.get_context(), Queue.get_device()};

  const size_t N = 10;
  float *Arr = malloc_shared<float>(N, Queue);

  // Buffer elements set to 0.5
  auto E1 = Queue.submit([&](handler &CGH) {
    CGH.parallel_for(range<1>{N}, [=](id<1> idx) {
      size_t i = idx;
      Arr[i] = 0.5f;
    });
  });

  Graph.add([&](handler &CGH) {
    CGH.parallel_for(range<1>{N}, [=](id<1> idx) {
      size_t i = idx;
      Arr[i] += 0.25f;
    });
  });

  // Buffer elements set to 1.5
  auto E2 = Queue.submit([&](handler &CGH) {
    CGH.depends_on(E1);
    CGH.parallel_for(range<1>{N}, [=](id<1> idx) {
      size_t i = idx;
      Arr[i] += 1.0f;
    });
  });

  auto ExecGraph = Graph.finalize();

  // Buffer elements set to 3.0
  auto E3 = Queue.submit([&](handler &CGH) {
    CGH.depends_on(E2);
    CGH.parallel_for(range<1>{N}, [=](id<1> idx) {
      size_t i = idx;
      Arr[i] *= 2.0f;
    });
  });

  // Buffer elements set to 3.25
  auto E4 = Queue.submit([&](handler &CGH) {
    CGH.depends_on(E3);
    CGH.ext_oneapi_graph(ExecGraph);
  });

  // Buffer elements set to 6.5
  auto E5 = Queue.submit([&](handler &CGH) {
    CGH.depends_on(E4);
    CGH.parallel_for(range<1>{N}, [=](id<1> idx) {
      size_t i = idx;
      Arr[i] *= 2.0f;
    });
  });

  // Buffer elements set to 6.75
  Queue.submit([&](handler &CGH) {
    CGH.depends_on(E5);
    CGH.ext_oneapi_graph(ExecGraph);
  });

  Queue.wait();

  for (size_t i = 0; i < N; i++) {
    assert(Arr[i] == 6.75f);
  }

  return 0;
}
