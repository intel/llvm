// RUN: %{build} -o %t.out
// RUN: %{run} %t.out

// Tests submitting memset to an in-order queue before recording
// commands from it.

#include "../graph_common.hpp"

int main() {
  using T = int;

  queue Queue{{sycl::ext::intel::property::queue::no_immediate_command_list{},
               sycl::property::queue::in_order{}}};

  if (!are_graphs_supported(Queue)) {
    return 0;
  }

  // Check if device has usm shared allocation
  if (!Queue.get_device().has(sycl::aspect::usm_shared_allocations))
    return 0;

  T *TestData = sycl::malloc_shared<T>(Size, Queue);
  T *TestDataOut = sycl::malloc_shared<T>(Size, Queue);

  ext::oneapi::experimental::command_graph Graph{Queue.get_context(),
                                                 Queue.get_device()};

  Queue.memset(TestData, 1, Size * sizeof(T));

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

  std::vector<T> Reference(Size);
  std::memset(Reference.data(), 1, Size * sizeof(T));
  for (size_t i = 0; i < Size; i++) {
    Reference[i] += i;
  }

  // Check Outputs
  for (size_t i = 0; i < Size; i++) {
    assert(TestDataOut[i] == Reference[i]);
  }

  sycl::free(TestData, Queue);

  return 0;
}
