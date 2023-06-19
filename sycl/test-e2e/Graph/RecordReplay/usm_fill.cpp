// REQUIRES: level_zero, gpu
// RUN: %{build} -o %t.out
// RUN: %{run} %t.out
// RUN: %if ext_oneapi_level_zero %{env ZE_DEBUG=4 %{run} %t.out 2>&1 | FileCheck %s %}
//
// CHECK-NOT: LEAK

// Tests a handler USM fill operation begin used as a graph node.
// The second run is to check that there are no leaks reported with the embedded
// ZE_DEBUG=4 testing capability.

#include "../graph_common.hpp"

int main() {

  queue Queue{gpu_selector_v};

  exp_ext::command_graph Graph{Queue.get_context(), Queue.get_device()};

  const size_t N = 10;
  float *Arr = malloc_device<float>(N, Queue);

  Graph.begin_recording(Queue);
  float PatternA = 1.0f;
  auto EventA = Queue.fill(Arr, PatternA, N);
  Graph.end_recording(Queue);
  auto ExecGraphA = Graph.finalize();

  Graph.begin_recording(Queue);
  float PatternB = 2.0f;
  auto EventB = Queue.fill(Arr, PatternB, N, EventA);
  Graph.end_recording(Queue);
  auto ExecGraphB = Graph.finalize();

  Graph.begin_recording(Queue);
  float PatternC = 3.0f;
  auto EventC = Queue.fill(Arr, PatternC, N, {EventA, EventB});
  Graph.end_recording(Queue);
  auto ExecGraphC = Graph.finalize();

  Graph.begin_recording(Queue);
  float PatternD = 3.14f;
  Queue.submit([&](handler &CGH) {
    CGH.depends_on(EventC);
    CGH.fill(Arr, PatternD, N);
  });
  Graph.end_recording(Queue);
  auto ExecGraphD = Graph.finalize();

  auto verifyLambda =
      [&](exp_ext::command_graph<exp_ext::graph_state::executable> ExecGraph,
          float Pattern) {
        Queue.submit([&](handler &CGH) { CGH.ext_oneapi_graph(ExecGraph); })
            .wait();
        std::vector<float> Output(N);
        Queue.memcpy(Output.data(), Arr, N * sizeof(float)).wait();

        for (int i = 0; i < N; i++)
          assert(Output[i] == Pattern);
      };

  verifyLambda(ExecGraphA, PatternA);
  verifyLambda(ExecGraphB, PatternB);
  verifyLambda(ExecGraphC, PatternC);
  verifyLambda(ExecGraphD, PatternD);

  sycl::free(Arr, Queue);

  return 0;
}
