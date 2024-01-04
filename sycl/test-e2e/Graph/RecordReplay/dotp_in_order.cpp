// RUN: %{build} -o %t.out
// RUN: %{run} %t.out
// Extra run to check for leaks in Level Zero using UR_L0_LEAKS_DEBUG
// RUN: %if level_zero %{env UR_L0_LEAKS_DEBUG=1 %{run} %t.out 2>&1 | FileCheck %s %}
//
// CHECK-NOT: LEAK

// Tests a dotp operation using device USM and an in-order queue.

#include "../graph_common.hpp"

int main() {
  property_list Properties{
      property::queue::in_order{},
      sycl::ext::intel::property::queue::no_immediate_command_list{}};
  queue Queue{Properties};

  if (!are_graphs_supported(Queue)) {
    return 0;
  }

  exp_ext::command_graph Graph{Queue.get_context(), Queue.get_device()};

  int *Dotp = malloc_device<int>(1, Queue);
  Queue.memset(Dotp, 0, sizeof(int)).wait();

  const size_t N = 10;
  int *X = malloc_device<int>(N, Queue);
  int *Y = malloc_device<int>(N, Queue);
  int *Z = malloc_device<int>(N, Queue);

  Graph.begin_recording(Queue);

  auto InitEvent = Queue.submit([&](handler &CGH) {
    CGH.parallel_for(N, [=](id<1> it) {
      X[it] = 1;
      Y[it] = 2;
      Z[it] = 3;
    });
  });

  auto EventA = Queue.submit([&](handler &CGH) {
    CGH.parallel_for(range<1>{N},
                     [=](id<1> it) { X[it] = Alpha * X[it] + Beta * Y[it]; });
  });

  auto EventB = Queue.submit([&](handler &CGH) {
    CGH.parallel_for(range<1>{N},
                     [=](id<1> it) { Z[it] = Gamma * Z[it] + Beta * Y[it]; });
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

  int Output;
  Queue.memcpy(&Output, Dotp, sizeof(int)).wait();

  assert(Output == dotp_reference_result(N));

  sycl::free(Dotp, Queue);
  sycl::free(X, Queue);
  sycl::free(Y, Queue);
  sycl::free(Z, Queue);

  return 0;
}
