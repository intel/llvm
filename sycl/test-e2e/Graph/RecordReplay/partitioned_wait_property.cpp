// RUN: %{build} -o %t.out
// RUN: %{run} %t.out

// Tests that without allow_wait_recording, queue::wait() during recording still
// throws, and that with the property it succeeds.

#include "../graph_common.hpp"

#include <sycl/properties/all_properties.hpp>

int main() {
  queue Queue{property::queue::in_order{}};

  // Test 1: Without property, queue::wait() should throw
  {
    exp_ext::command_graph Graph{Queue.get_context(), Queue.get_device()};
    Graph.begin_recording(Queue);

    std::error_code ErrorCode = make_error_code(sycl::errc::success);
    try {
      Queue.wait();
    } catch (const sycl::exception &e) {
      ErrorCode = e.code();
    }
    assert(ErrorCode == sycl::errc::invalid);

    Graph.end_recording(Queue);
  }

  // Test 2: With property, queue::wait() should succeed
  {
    exp_ext::command_graph Graph{
        Queue.get_context(), Queue.get_device(),
        {exp_ext::property::graph::allow_wait_recording{}}};
    Graph.begin_recording(Queue);

    bool ExceptionThrown = false;
    try {
      Queue.wait();
    } catch (const sycl::exception &) {
      ExceptionThrown = true;
    }
    assert(!ExceptionThrown);

    Graph.end_recording(Queue);
  }

  // Test 3: Without property, event::wait() should throw
  {
    exp_ext::command_graph Graph{Queue.get_context(), Queue.get_device()};
    Graph.begin_recording(Queue);

    auto Event = Queue.submit(
        [&](handler &CGH) { CGH.single_task<class TestKernel1>([=]() {}); });

    Graph.end_recording(Queue);

    std::error_code ErrorCode = make_error_code(sycl::errc::success);
    try {
      Event.wait();
    } catch (const sycl::exception &e) {
      ErrorCode = e.code();
    }
    assert(ErrorCode == sycl::errc::invalid);
  }

  // Test 4: With property, event::wait() should succeed
  {
    exp_ext::command_graph Graph{
        Queue.get_context(), Queue.get_device(),
        {exp_ext::property::graph::allow_wait_recording{}}};
    Graph.begin_recording(Queue);

    auto Event = Queue.submit(
        [&](handler &CGH) { CGH.single_task<class TestKernel2>([=]() {}); });

    // Note: The event is associated with the graph, so wait() will create
    // a host_sync node instead of throwing
    bool ExceptionThrown = false;
    try {
      Event.wait();
    } catch (const sycl::exception &) {
      ExceptionThrown = true;
    }
    assert(!ExceptionThrown);

    Graph.end_recording(Queue);
  }

  return 0;
}
