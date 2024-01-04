// Tests that using a buffer in a graph will throw, unless the
// assume_buffer_outlives_graph property is passed on graph creation.

#include "../graph_common.hpp"

int main() {
  queue Queue;

  if (!are_graphs_supported(Queue)) {
    return 0;
  }

  using T = unsigned short;

  buffer<T> Buffer{range<1>{1}};
  Buffer.set_write_back(false);

  // Test with the property
  {
    exp_ext::command_graph Graph{
        Queue.get_context(),
        Queue.get_device(),
        {exp_ext::property::graph::assume_buffer_outlives_graph{}}};

    std::error_code ErrorCode = make_error_code(sycl::errc::success);
    // This should not throw because we have passed the property
    try {
      add_node(Graph, Queue, [&](handler &CGH) {
        auto acc = Buffer.get_access(CGH);
        CGH.single_task([=]() {});
      });
    } catch (const sycl::exception &e) {
      ErrorCode = e.code();
    }
    assert(ErrorCode == sycl::errc::success);
  }

  // Test without the property
  {
    exp_ext::command_graph Graph{Queue.get_context(), Queue.get_device()};

    std::error_code ErrorCode = make_error_code(sycl::errc::success);
    // This should throw because we have not passed the property
    try {
      add_node(Graph, Queue, [&](handler &CGH) {
        auto acc = Buffer.get_access(CGH);
        CGH.single_task([=]() {});
      });
    } catch (const sycl::exception &e) {
      ErrorCode = e.code();
    }
    assert(ErrorCode == sycl::errc::invalid);
  }

  return 0;
}
