// RUN: %{build} -o %t.out
// RUN: %{run} %t.out
// Extra run to check for leaks in Level Zero using UR_L0_LEAKS_DEBUG
// RUN: %if (level_zero && linux) %{env UR_L0_LEAKS_DEBUG=1 %{run} %t.out 2>&1 | FileCheck %s %}
// RUN: %if (level_zero && windows) %{env UR_L0_LEAKS_DEBUG=1 env SYCL_ENABLE_DEFAULT_CONTEXTS=0 %{run} %t.out 2>&1 | FileCheck %s %}
//
// CHECK-NOT: LEAK

// Test submitting the same graph twice with another command in between, this
// intermediate command depends on the first submission of the graph, and
// is a dependency of the second submission of the graph.

#include "../graph_common.hpp"
int main() {
  queue Queue{{sycl::ext::intel::property::queue::no_immediate_command_list{}}};

  if (!are_graphs_supported(Queue)) {
    return 0;
  }

  exp_ext::command_graph Graph{Queue.get_context(), Queue.get_device()};

  const size_t N = 10;
  int *Arr = malloc_shared<int>(N, Queue);

  // Buffer elements set to 3
  auto E1 = Queue.submit([&](handler &CGH) {
    CGH.parallel_for(range<1>{N}, [=](id<1> idx) {
      size_t i = idx;
      Arr[i] = 3;
    });
  });

  Graph.add([&](handler &CGH) {
    CGH.parallel_for(range<1>{N}, [=](id<1> idx) {
      size_t i = idx;
      Arr[i] += 2;
    });
  });

  // Buffer elements set to 4
  auto E2 = Queue.submit([&](handler &CGH) {
    CGH.depends_on(E1);
    CGH.parallel_for(range<1>{N}, [=](id<1> idx) {
      size_t i = idx;
      Arr[i] += 1;
    });
  });

  auto ExecGraph = Graph.finalize();

  // Buffer elements set to 8
  auto E3 = Queue.submit([&](handler &CGH) {
    CGH.depends_on(E2);
    CGH.parallel_for(range<1>{N}, [=](id<1> idx) {
      size_t i = idx;
      Arr[i] *= 2;
    });
  });

  // Buffer elements set to 10
  auto E4 = Queue.submit([&](handler &CGH) {
    CGH.depends_on(E3);
    CGH.ext_oneapi_graph(ExecGraph);
  });

  // Buffer elements set to 20
  auto E5 = Queue.submit([&](handler &CGH) {
    CGH.depends_on(E4);
    CGH.parallel_for(range<1>{N}, [=](id<1> idx) {
      size_t i = idx;
      Arr[i] *= 2;
    });
  });

  // Buffer elements set to 22
  Queue.submit([&](handler &CGH) {
    CGH.depends_on(E5);
    CGH.ext_oneapi_graph(ExecGraph);
  });

  Queue.wait();

  const int Expected = 22;
  for (size_t i = 0; i < N; i++) {
    assert(check_value(i, Expected, Arr[i], "Arr"));
  }

  // Free the allocated memory
  sycl::free(Arr, Queue);

  return 0;
}
