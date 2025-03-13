// RUN: %{build} -o %t.out
// RUN: %{run} %t.out
// RUN: %if preview-breaking-changes-supported %{ %{build} -fpreview-breaking-changes -o %t2.out %}
// RUN: %if preview-breaking-changes-supported %{ %{run} %t2.out %}
// REQUIRES: cuda

// Test that interop_handle::ext_codeplay_get_native_graph() throws if no
// backend graph object is available.

#include <sycl/backend.hpp>
#include <sycl/ext/oneapi/experimental/graph.hpp>
#include <sycl/interop_handle.hpp>

namespace exp_ext = sycl::ext::oneapi::experimental;
using namespace sycl;

// Eager queue native command submissions have no graph attached. The native
// command function object is processed immediately, so we don't need a queue
// wait_and_throw call to see the error.
void getNativeGraphOnEagerQueue(queue Queue) {
  std::error_code ExceptionCode = make_error_code(sycl::errc::success);
  try {
    Queue.submit([&](sycl::handler &CGH) {
      CGH.ext_codeplay_enqueue_native_command([=](sycl::interop_handle IH) {
        assert(false == IH.ext_codeplay_has_graph());
        auto NativeGraph =
            IH.ext_codeplay_get_native_graph<sycl::backend::ext_oneapi_cuda>();
        (void)NativeGraph;
      });
    });
  } catch (exception &Exception) {
    ExceptionCode = Exception.code();
  }
  assert(ExceptionCode != sycl::errc::success);
}

// Eager queue host-task submissions have no graph attached. The host-task
// object is processed asynchronously respecting command-group dependencies,
// so need a queue wait_and_throw call to see the error.
void getNativeGraphInEagerHostTask(queue Queue) {
  std::error_code ExceptionCode = make_error_code(sycl::errc::success);
  try {
    Queue.submit([&](sycl::handler &CGH) {
      auto Func = [=](interop_handle IH) {
        assert(false == IH.ext_codeplay_has_graph());
        auto NativeGraph =
            IH.ext_codeplay_get_native_graph<sycl::backend::ext_oneapi_cuda>();
        (void)NativeGraph;
      };

      CGH.host_task(Func);
    });
    Queue.wait_and_throw();
  } catch (exception &Exception) {
    ExceptionCode = Exception.code();
  }
  assert(ExceptionCode != sycl::errc::success);
}

// host-task submissions recorded from a queue to a graph have no backend graph
// attached, as the graph is partitioned into multiple backend graph objects
// around the host task. The host-task function object is only processed on
// graph execution, respecting node dependencies, so we need need to finalize
// the graph and submit to a queue before the user can see the error.
void getNativeGraphInGraphHostTask(queue Queue) {
  exp_ext::command_graph Graph{Queue};
  std::error_code ExceptionCode = make_error_code(sycl::errc::success);
  try {
    Graph.add([&](sycl::handler &CGH) {
      auto Func = [=](interop_handle IH) {
        assert(false == IH.ext_codeplay_has_graph());
        auto NativeGraph =
            IH.ext_codeplay_get_native_graph<sycl::backend::ext_oneapi_cuda>();
        (void)NativeGraph;
      };

      CGH.host_task(Func);
    });
    auto ExecGraph = Graph.finalize();
    Queue.ext_oneapi_graph(ExecGraph);
    Queue.wait_and_throw();
  } catch (exception &Exception) {
    ExceptionCode = Exception.code();
  }
  assert(ExceptionCode != sycl::errc::success);
}

int main() {
  queue Queue([](sycl::exception_list ExceptionList) {
    std::rethrow_exception(*ExceptionList.begin());
  });

  getNativeGraphOnEagerQueue(Queue);
  getNativeGraphInEagerHostTask(Queue);
  getNativeGraphInGraphHostTask(Queue);

  return 0;
}
