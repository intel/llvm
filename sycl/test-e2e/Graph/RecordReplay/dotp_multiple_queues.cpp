// REQUIRES: level_zero, gpu
// RUN: %{build} -o %t.out
// RUN: %{run} %t.out
// RUN: %if ext_oneapi_level_zero %{env ZE_DEBUG=4 %{run} %t.out 2>&1 | FileCheck %s %}
//
// CHECK-NOT: LEAK

// Tests a dotp operation split between 2 in-order queues using device USM.

#include "../graph_common.hpp"

int main() {

  property_list properties{property::queue::in_order()};
  queue QueueA{gpu_selector_v, properties};
  queue QueueB{gpu_selector_v, properties};

  exp_ext::command_graph Graph{QueueA.get_context(), QueueA.get_device()};

  float *Dotp = malloc_device<float>(1, QueueA);

  const size_t N = 10;
  float *X = malloc_device<float>(N, QueueA);
  float *Y = malloc_device<float>(N, QueueA);
  float *Z = malloc_device<float>(N, QueueA);

  Graph.begin_recording(QueueA);
  Graph.begin_recording(QueueB);

  QueueA.submit([&](handler &CGH) {
    CGH.parallel_for(N, [=](id<1> it) {
      const size_t i = it[0];
      X[i] = 1.0f;
      Y[i] = 2.0f;
      Z[i] = 3.0f;
    });
  });

  auto Event = QueueA.submit([&](handler &CGH) {
    CGH.parallel_for(range<1>{N}, [=](id<1> it) {
      const size_t i = it[0];
      X[i] = Alpha * X[i] + Beta * Y[i];
    });
  });

  QueueB.submit([&](handler &CGH) {
    CGH.depends_on(Event); // needed for cross queue dependency
    CGH.parallel_for(range<1>{N}, [=](id<1> it) {
      const size_t i = it[0];
      Z[i] = Gamma * Z[i] + Beta * Y[i];
    });
  });

  QueueB.submit([&](handler &CGH) {
    CGH.single_task([=]() {
      for (size_t j = 0; j < N; j++) {
        Dotp[0] += X[j] * Z[j];
      }
    });
  });

  Graph.end_recording();

  auto ExecGraph = Graph.finalize();

  QueueA.submit([&](handler &CGH) { CGH.ext_oneapi_graph(ExecGraph); });

  float Output;
  QueueA.memcpy(&Output, Dotp, sizeof(float)).wait();

  assert(Output == dotp_reference_result(N));

  sycl::free(Dotp, QueueA);
  sycl::free(X, QueueA);
  sycl::free(Y, QueueA);
  sycl::free(Z, QueueA);

  return 0;
}
