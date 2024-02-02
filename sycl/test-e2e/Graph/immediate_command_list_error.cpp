// REQUIRES: level_zero, gpu
// RUN: %{build} -o %t.out
// RUN: %{run} %t.out
// RUN: %if level_zero  %{ %{l0_leak_check} %{run} %t.out 2>&1 | FileCheck %s %}
//
// CHECK-NOT: LEAK

// Tests that graph submission will throw if the target queue is using immediate
// command lists and not throw if they are using regular command queues.

#include "graph_common.hpp"

int main() {
  queue QueueImmediate{
      {sycl::ext::intel::property::queue::immediate_command_list{}}};
  queue QueueNoImmediate{
      QueueImmediate.get_context(),
      QueueImmediate.get_device(),
      {sycl::ext::intel::property::queue::no_immediate_command_list{}}};

  if (!are_graphs_supported(QueueNoImmediate)) {
    return 0;
  }

  exp_ext::command_graph Graph{QueueNoImmediate.get_context(),
                               QueueNoImmediate.get_device()};

  std::error_code ErrorCode = make_error_code(sycl::errc::success);
  try {
    auto GraphExec = Graph.finalize();
    QueueNoImmediate.submit(
        [&](handler &CGH) { CGH.ext_oneapi_graph(GraphExec); });
  } catch (sycl::exception &E) {
    ErrorCode = E.code();
  }

  assert(ErrorCode == make_error_code(errc::success));

  ErrorCode = make_error_code(sycl::errc::success);
  try {
    auto GraphExec = Graph.finalize();
    QueueImmediate.submit(
        [&](handler &CGH) { CGH.ext_oneapi_graph(GraphExec); });
  } catch (sycl::exception &E) {
    ErrorCode = E.code();
  }

  assert(ErrorCode == make_error_code(errc::invalid));

  return 0;
}
