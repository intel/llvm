// REQUIRES: level_zero, gpu
// RUN: %{build} -o %t.out
// RUN: %{run} %t.out
// RUN: %if ext_oneapi_level_zero %{env ZE_DEBUG=4 %{run} %t.out 2>&1 | FileCheck %s %}
//
// CHECK-NOT: LEAK


// Tests that buffer accessors exhibit the correct behaviour when:
// * A node is added to the graph between two queue submissions which
//   use the same buffer, but are not added to the graph.
//
// * A queue submission using the same buffer is made after finalization
//   of the graph, but before graph execution.
//
// * The graph is submitted for execution twice separated by a queue
//   submission using the same buffer, this should respect dependencies and
//   create the correct ordering.
// The second run is to check that there are no leaks reported with the embedded
// ZE_DEBUG=4 testing capability.

#include "../graph_common.hpp"
int main() {

  queue Queue{gpu_selector_v};

  exp_ext::command_graph Graph{Queue.get_context(), Queue.get_device()};

  const size_t N = 10;
  std::vector<float> Arr(N, 0.0f);

  buffer<float> Buf{N};
  Buf.set_write_back(false);

  // Buffer elements set to 0.5
  Queue.submit([&](handler &CGH) {
    auto Acc = Buf.get_access(CGH);
    CGH.parallel_for(range<1>{N}, [=](id<1> idx) {
      size_t i = idx;
      Acc[i] = 0.5f;
    });
  });

  Graph.add([&](handler &CGH) {
    auto Acc = Buf.get_access(CGH);
    CGH.parallel_for(range<1>{N}, [=](id<1> idx) {
      size_t i = idx;
      Acc[i] += 0.25f;
    });
  });

  for (size_t i = 0; i < N; i++) {
    assert(Arr[i] == 0.0f);
  }

  // Buffer elements set to 1.5
  Queue.submit([&](handler &CGH) {
    auto Acc = Buf.get_access(CGH);
    CGH.parallel_for(range<1>{N}, [=](id<1> idx) {
      size_t i = idx;
      Acc[i] += 1.0f;
    });
  });

  auto ExecGraph = Graph.finalize();

  for (size_t i = 0; i < N; i++) {
    assert(Arr[i] == 0.0f);
  }

  // Buffer elements set to 3.0
  Queue.submit([&](handler &CGH) {
    auto Acc = Buf.get_access(CGH);
    CGH.parallel_for(range<1>{N}, [=](id<1> idx) {
      size_t i = idx;
      Acc[i] *= 2.0f;
    });
  });

  // Buffer elements set to 3.25
  Queue.submit([&](handler &CGH) { CGH.ext_oneapi_graph(ExecGraph); });

  // Buffer elements set to 6.5
  Queue.submit([&](handler &CGH) {
    auto Acc = Buf.get_access(CGH);
    CGH.parallel_for(range<1>{N}, [=](id<1> idx) {
      size_t i = idx;
      Acc[i] *= 2.0f;
    });
  });

  // Buffer elements set to 6.75
  Queue.submit([&](handler &CGH) { CGH.ext_oneapi_graph(ExecGraph); });

  Queue.submit([&](handler &CGH) {
    auto Acc = Buf.get_access(CGH);
    CGH.copy(Acc, Arr.data());
  });
  Queue.wait();

  for (size_t i = 0; i < N; i++) {
    assert(Arr[i] == 6.75f);
  }

  return 0;
}
