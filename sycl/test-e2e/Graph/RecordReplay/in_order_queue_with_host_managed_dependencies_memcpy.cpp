// RUN: %{build} -o %t.out
// RUN: %{run} %t.out

// REQUIRES: aspect-usm_shared_allocations

// Tests submitting memcpy to an in-order queue before recording
// commands from it.

#include "../graph_common.hpp"

int main() {
  using T = int;

  queue Queue{sycl::property::queue::in_order{}};

  std::vector<T> TestDataIn(Size);
  T *TestData = sycl::malloc_shared<T>(Size, Queue);
  T *TestDataOut = sycl::malloc_shared<T>(Size, Queue);

  ext::oneapi::experimental::command_graph Graph{Queue.get_context(),
                                                 Queue.get_device()};

  std::memset(TestDataIn.data(), 1, Size * sizeof(T));
  Queue.memcpy(TestData, TestDataIn.data(), Size * sizeof(T));

  Graph.begin_recording(Queue);

  auto GraphEvent = Queue.submit([&](handler &CGH) {
    CGH.single_task<class TestKernel2>([=]() {
      for (size_t i = 0; i < Size; i++) {
        TestData[i] += static_cast<T>(i);
      }
    });
  });

  Graph.end_recording(Queue);

  auto GraphExec = Graph.finalize();
  Queue.submit([&](handler &CGH) { CGH.ext_oneapi_graph(GraphExec); });

  Queue.memcpy(TestDataOut, TestData, Size * sizeof(T));

  Queue.wait_and_throw();

  // Check Outputs
  T Reference = 0;
  std::memset(&Reference, 1, sizeof(T));
  for (size_t i = 0; i < Size; i++) {
    assert(TestDataOut[i] == (Reference + i));
  }

  sycl::free(TestData, Queue);
  sycl::free(TestDataOut, Queue);

  return 0;
}
