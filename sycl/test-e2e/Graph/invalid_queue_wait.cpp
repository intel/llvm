// RUN: %{build} -o %t.out
// RUN: %{run} %t.out

// Tests that waiting on a Queue in recording mode throws.

#include "graph_common.hpp"

int main() {
  queue Queue{};

  if (!are_graphs_supported(Queue)) {
    return 0;
  }

  ext::oneapi::experimental::command_graph Graph{Queue.get_context(),
                                                 Queue.get_device()};
  Graph.begin_recording(Queue);

  std::error_code ErrorCode = make_error_code(sycl::errc::success);

  try {
    Queue.wait();
  } catch (const sycl::exception &e) {
    ErrorCode = e.code();
  }
  assert(ErrorCode == sycl::errc::invalid);

  return 0;
}
