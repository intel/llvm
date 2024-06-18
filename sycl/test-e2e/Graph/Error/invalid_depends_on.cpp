// RUN: %{build} -o %t.out
// RUN: %{run} %t.out

// Tests that calling handler::depends_on() for events not part of the graph
// throws.

#include "../graph_common.hpp"

int main() {
  queue Queue{};

  ext::oneapi::experimental::command_graph Graph{Queue.get_context(),
                                                 Queue.get_device()};
  ext::oneapi::experimental::command_graph Graph2{Queue.get_context(),
                                                  Queue.get_device()};

  auto NormalEvent = Queue.submit(
      [&](handler &CGH) { CGH.single_task<class TestKernel1>([=]() {}); });

  Graph2.begin_recording(Queue);

  auto OtherGraphEvent = Queue.submit(
      [&](handler &CGH) { CGH.single_task<class TestKernel2>([=]() {}); });

  Graph2.end_recording(Queue);

  Graph.begin_recording(Queue);

  // Test that depends_on in explicit and record and replay throws from an event
  // outside any graph.

  std::error_code ErrorCode = make_error_code(sycl::errc::success);
  try {
    auto GraphEvent = Queue.submit([&](handler &CGH) {
      CGH.depends_on(NormalEvent);
      CGH.single_task<class TestKernel3>([=]() {});
    });
  } catch (const sycl::exception &e) {
    ErrorCode = e.code();
  }
  assert(ErrorCode == sycl::errc::invalid);

  ErrorCode = make_error_code(sycl::errc::success);
  try {
    Graph.add([&](handler &CGH) {
      CGH.depends_on(NormalEvent);
      CGH.single_task<class TestKernel4>([=]() {});
    });
  } catch (const sycl::exception &e) {
    ErrorCode = e.code();
  }
  assert(ErrorCode == sycl::errc::invalid);

  // Test that depends_on throws from an event from another graph.
  ErrorCode = make_error_code(sycl::errc::success);
  try {
    auto GraphEvent = Queue.submit([&](handler &CGH) {
      CGH.depends_on(OtherGraphEvent);
      CGH.single_task<class TestKernel5>([=]() {});
    });
  } catch (const sycl::exception &e) {
    ErrorCode = e.code();
  }
  assert(ErrorCode == sycl::errc::invalid);

  ErrorCode = make_error_code(sycl::errc::success);
  try {
    Graph.add([&](handler &CGH) {
      CGH.depends_on(OtherGraphEvent);
      CGH.single_task<class TestKernel6>([=]() {});
    });
  } catch (const sycl::exception &e) {
    ErrorCode = e.code();
  }
  assert(ErrorCode == sycl::errc::invalid);

  return 0;
}
