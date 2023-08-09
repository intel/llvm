// REQUIRES: cuda, gpu
// RUN: %{build} -o %t.out
// RUN: %{run} %t.out

// Tests the ability to finalize a empty command graph
// without submitting the graph.

#include "graph_common.hpp"

int GetCudaBackend(const sycl::device &Dev) {
  // Return 1 if the device backend is "cuda" or 0 else.
  // 0 does not prevent another device to be picked as a second choice
  return Dev.get_backend() == backend::ext_oneapi_cuda;
}

int main() {
  sycl::device CudaDev{GetCudaBackend};
  queue Queue{CudaDev};

  // Skip the test if no cuda backend found
  if (CudaDev.get_backend() != backend::ext_oneapi_cuda)
    return 0;

  std::error_code ErrorCode = make_error_code(sycl::errc::success);
  // This should not throw an exception
  try {
    exp_ext::command_graph Graph{Queue.get_context(), CudaDev};
    auto GraphExec = Graph.finalize();
  } catch (const sycl::exception &e) {
    ErrorCode = e.code();
  }
  assert(ErrorCode == sycl::errc::success);

  return 0;
}
