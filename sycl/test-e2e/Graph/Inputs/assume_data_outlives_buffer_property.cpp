// Tests that using a buffer which is created with a host pointer in a graph
// will throw, unless the assume_data_outlives_buffer property is passed on
// graph creation.

#include "../graph_common.hpp"

int main() {
  queue Queue;

  using T = unsigned short;

  T Data = 0;

  buffer<T> BufferHost{&Data, range<1>{1}};
  BufferHost.set_write_back(false);
  buffer<T> BufferNoHost{range<1>{1}};
  BufferNoHost.set_write_back(false);

  // Test with the property
  {
    exp_ext::command_graph Graph{
        Queue.get_context(),
        Queue.get_device(),
        {exp_ext::property::graph::assume_buffer_outlives_graph{},
         exp_ext::property::graph::assume_data_outlives_buffer{}}};

    std::error_code ErrorCode = make_error_code(sycl::errc::success);
    // This should not throw because we have passed the property
    try {
      add_node(Graph, Queue, [&](handler &CGH) {
        auto acc = BufferHost.get_access(CGH);
        CGH.single_task([=]() {});
      });
    } catch (const sycl::exception &e) {
      ErrorCode = e.code();
    }
    assert(ErrorCode == sycl::errc::success);

    // This should not throw regardless of property use
    try {
      add_node(Graph, Queue, [&](handler &CGH) {
        auto acc = BufferNoHost.get_access(CGH);
        CGH.single_task([=]() {});
      });
    } catch (const sycl::exception &e) {
      ErrorCode = e.code();
    }
    assert(ErrorCode == sycl::errc::success);
  }

  // Test without the property
  {
    exp_ext::command_graph Graph{
        Queue.get_context(),
        Queue.get_device(),
        {exp_ext::property::graph::assume_buffer_outlives_graph{}}};

    std::error_code ErrorCode = make_error_code(sycl::errc::success);
    // This should throw because we haven't used the property
    try {
      add_node(Graph, Queue, [&](handler &CGH) {
        auto acc = BufferHost.get_access(CGH);
        CGH.single_task([=]() {});
      });
    } catch (const sycl::exception &e) {
      ErrorCode = e.code();
    }
    assert(ErrorCode == sycl::errc::invalid);

    ErrorCode = sycl::errc::success;
    // This should not throw regardless of property use
    try {
      add_node(Graph, Queue, [&](handler &CGH) {
        auto acc = BufferNoHost.get_access(CGH);
        CGH.single_task([=]() {});
      });
    } catch (const sycl::exception &e) {
      ErrorCode = e.code();
    }
    assert(ErrorCode == sycl::errc::success);
  }

  return 0;
}
