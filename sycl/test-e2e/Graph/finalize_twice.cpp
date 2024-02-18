// RUN: %{build} -o %t.out
// RUN: %{run} %t.out

// Tests calling finalize() more than once on the same command_graph.

#include "graph_common.hpp"

int main() {
  queue Queue{};

  if (!are_graphs_supported(Queue)) {
    return 0;
  }

  ext::oneapi::experimental::command_graph Graph{Queue.get_context(),
                                                 Queue.get_device()};
  auto GraphExec = Graph.finalize();

  std::error_code ErrorCode = make_error_code(sycl::errc::success);
  try {
    auto GraphExec2 = Graph.finalize();
  } catch (const sycl::exception &e) {
    ErrorCode = e.code();
  }
  assert(ErrorCode == sycl::errc::success);

  return 0;
}
