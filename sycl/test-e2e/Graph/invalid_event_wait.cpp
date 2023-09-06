// REQUIRES: cuda || level_zero, gpu
// RUN: %{build} -o %t.out
// RUN: %{run} %t.out

// Tests that waiting on an event returned from a Record and Replay submission
// throws.

#include "graph_common.hpp"

int main() {
  queue Queue{{sycl::ext::intel::property::queue::no_immediate_command_list{}}};

  ext::oneapi::experimental::command_graph Graph{Queue.get_context(),
                                                 Queue.get_device()};
  Graph.begin_recording(Queue);

  auto GraphEvent = Queue.submit(
      [&](handler &CGH) { CGH.single_task<class TestKernel>([=]() {}); });

  Graph.end_recording(Queue);

  std::error_code ErrorCode = make_error_code(sycl::errc::success);
  try {
    GraphEvent.wait();
  } catch (const sycl::exception &e) {
    ErrorCode = e.code();
  }
  assert(ErrorCode == sycl::errc::invalid);

  return 0;
}
