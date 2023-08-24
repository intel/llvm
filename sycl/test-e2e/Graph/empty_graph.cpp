// REQUIRES: cuda || level_zero, gpu
// RUN: %{build} -o %t.out
// RUN: %{run} %t.out

// Tests the ability to finalize and submit a command graph which doesn't
// contain any nodes.

#include "graph_common.hpp"

int main() {
  queue Queue{{sycl::ext::intel::property::queue::no_immediate_command_list{}}};

  ext::oneapi::experimental::command_graph Graph{Queue.get_context(),
                                                 Queue.get_device()};

  std::error_code ErrorCode = make_error_code(sycl::errc::success);
  try {
    auto GraphExec = Graph.finalize();
    Queue.submit([&](handler &CGH) { CGH.ext_oneapi_graph(GraphExec); });
    Queue.wait_and_throw();
  } catch (const sycl::exception &e) {
    ErrorCode = e.code();
  }
  assert(ErrorCode == sycl::errc::success);

  return 0;
}
