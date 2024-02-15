// RUN: %{build} -o %t.out
// RUN: %{run} %t.out
// Extra run to check for leaks in Level Zero using UR_L0_LEAKS_DEBUG
// RUN: %if level_zero %{env SYCL_PI_LEVEL_ZERO_USE_IMMEDIATE_COMMANDLISTS=0 %{l0_leak_check} %{run} %t.out 2>&1 | FileCheck %s --implicit-check-not=LEAK %}
// Extra run to check for immediate-command-list in Level Zero
// RUN: %if level_zero && linux %{env SYCL_PI_LEVEL_ZERO_USE_IMMEDIATE_COMMANDLISTS=1 %{l0_leak_check} %{run} %t.out 2>&1 | FileCheck %s --implicit-check-not=LEAK %}
//

// Tests a dotp operation split between 2 in-order queues using device USM.

#include "../graph_common.hpp"

int main() {
  property_list Properties{
      property::queue::in_order{},
      sycl::ext::intel::property::queue::no_immediate_command_list{}};
  queue QueueA{Properties};

  if (!are_graphs_supported(QueueA)) {
    return 0;
  }

  queue QueueB{QueueA.get_context(), QueueA.get_device(), Properties};

  exp_ext::command_graph Graph{QueueA.get_context(), QueueA.get_device()};

  int *Dotp = malloc_device<int>(1, QueueA);
  QueueA.memset(Dotp, 0, sizeof(int)).wait();

  const size_t N = 10;
  int *X = malloc_device<int>(N, QueueA);
  int *Y = malloc_device<int>(N, QueueA);
  int *Z = malloc_device<int>(N, QueueA);

  Graph.begin_recording(QueueA);
  Graph.begin_recording(QueueB);

  QueueA.submit([&](handler &CGH) {
    CGH.parallel_for(N, [=](id<1> it) {
      X[it] = 1;
      Y[it] = 2;
      Z[it] = 3;
    });
  });

  auto Event = QueueA.submit([&](handler &CGH) {
    CGH.parallel_for(range<1>{N},
                     [=](id<1> it) { X[it] = Alpha * X[it] + Beta * Y[it]; });
  });

  QueueB.submit([&](handler &CGH) {
    CGH.depends_on(Event); // needed for cross queue dependency
    CGH.parallel_for(range<1>{N},
                     [=](id<1> it) { Z[it] = Gamma * Z[it] + Beta * Y[it]; });
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

  int Output;
  QueueA.memcpy(&Output, Dotp, sizeof(int)).wait();

  assert(Output == dotp_reference_result(N));

  sycl::free(Dotp, QueueA);
  sycl::free(X, QueueA);
  sycl::free(Y, QueueA);
  sycl::free(Z, QueueA);

  return 0;
}
