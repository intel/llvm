// REQUIRES: level_zero, gpu
// RUN: %{build} -o %t.out
// RUN: %{run} %t.out
// RUN: %if ext_oneapi_level_zero %{env ZE_DEBUG=4 %{run} %t.out 2>&1 | FileCheck %s %}
//
// CHECK-NOT: LEAK

// Tests a dotp operation using device USM and an in-order queue.
// The second run is to check that there are no leaks reported with the embedded
// ZE_DEBUG=4 testing capability.

#include "../graph_common.hpp"

int main() {
  property_list properties{property::queue::in_order()};
  queue Queue{gpu_selector_v, properties};

  exp_ext::command_graph Graph{Queue.get_context(), Queue.get_device()};

  float *Dotp = malloc_device<float>(1, Queue);

  const size_t N = 10;
  float *X = malloc_device<float>(N, Queue);
  float *Y = malloc_device<float>(N, Queue);
  float *Z = malloc_device<float>(N, Queue);

  Graph.begin_recording(Queue);

  auto InitEvent = Queue.submit([&](handler &CGH) {
    CGH.parallel_for(N, [=](id<1> it) {
      const size_t i = it[0];
      X[i] = 1.0f;
      Y[i] = 2.0f;
      Z[i] = 3.0f;
    });
  });

  auto EventA = Queue.submit([&](handler &CGH) {
    CGH.parallel_for(range<1>{N}, [=](id<1> it) {
      const size_t i = it[0];
      X[i] = Alpha * X[i] + Beta * Y[i];
    });
  });

  auto EventB = Queue.submit([&](handler &CGH) {
    CGH.parallel_for(range<1>{N}, [=](id<1> it) {
      const size_t i = it[0];
      Z[i] = Gamma * Z[i] + Beta * Y[i];
    });
  });

  Queue.submit([&](handler &CGH) {
    CGH.single_task([=]() {
      for (size_t j = 0; j < N; j++) {
        Dotp[0] += X[j] * Z[j];
      }
    });
  });

  Graph.end_recording();

  auto ExecGraph = Graph.finalize();

  Queue.submit([&](handler &CGH) { CGH.ext_oneapi_graph(ExecGraph); });

  float Output;
  Queue.memcpy(&Output, Dotp, sizeof(float)).wait();

  assert(Output == dotp_reference_result(N));

  sycl::free(Dotp, Queue);
  sycl::free(X, Queue);
  sycl::free(Y, Queue);
  sycl::free(Z, Queue);

  return 0;
}
