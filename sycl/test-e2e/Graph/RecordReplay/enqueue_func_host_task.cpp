// REQUIRES: aspect-usm_shared_allocations

// RUN: %{build} -o %t.out
// RUN: %{run} %t.out
// RUN: %if level_zero %{%{l0_leak_check} %{run} %t.out 2>&1 | FileCheck %s --implicit-check-not=LEAK %}

// Tests that restricted host_task can be recorded via the command-buffer path

#include "../graph_common.hpp"

#include <sycl/ext/oneapi/experimental/enqueue_functions.hpp>
#include <sycl/properties/all_properties.hpp>

namespace syclex = sycl::ext::oneapi::experimental;

constexpr size_t N = 1024;

int main() {
  queue Queue{property::queue::in_order{}};

  const sycl::context Context = Queue.get_context();
  const sycl::device Device = Queue.get_device();

  uint32_t *Data = malloc_shared<uint32_t>(N, Queue);
  std::fill(Data, Data + N, 0);

  exp_ext::command_graph Graph{Context, Device};

  Graph.begin_recording(Queue);

  Queue.submit([&](handler &CGH) {
    CGH.parallel_for(range<1>{N}, [=](id<1> idx) {
      Data[idx] += static_cast<uint32_t>(idx[0]) + 1;
    });
  });

  syclex::host_task(Queue, [=] {
    for (size_t i = 0; i < N; i++) {
      Data[i] *= 2;
    }
  });

  Queue.submit([&](handler &CGH) {
    CGH.parallel_for(range<1>{N}, [=](id<1> idx) { Data[idx] += 10; });
  });

  Graph.end_recording(Queue);

  auto ExecutableGraph = Graph.finalize();

  Queue.submit([&](handler &CGH) { CGH.ext_oneapi_graph(ExecutableGraph); });
  Queue.wait();

  for (size_t i = 0; i < N; i++) {
    uint32_t Expected = static_cast<uint32_t>((i + 1) * 2 + 10);
    assert(check_value(i, Expected, Data[i], "Data"));
  }

  // Second execution
  Queue.submit([&](handler &CGH) { CGH.ext_oneapi_graph(ExecutableGraph); });
  Queue.wait();

  for (size_t i = 0; i < N; i++) {
    uint32_t Expected = static_cast<uint32_t>((i + 1) * 6 + 30);
    assert(check_value(i, Expected, Data[i], "Data"));
  }

  free(Data, Queue);
  return 0;
}
