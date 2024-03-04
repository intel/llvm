// RUN: %{build} -o %t.out
// RUN: %{run} %t.out

// Tests the using device query for graphs support, and that the return value
// matches expectations.

#include "../graph_common.hpp"

int main() {
  queue Queue;

  auto Device = Queue.get_device();
  bool SupportsGraphs = Device.has(aspect::ext_oneapi_graph);
  auto Backend = Device.get_backend();

  if ((Backend == backend::ext_oneapi_level_zero) ||
      (Backend == backend::ext_oneapi_cuda) ||
      (Backend == backend::ext_oneapi_hip)) {
    assert(SupportsGraphs);
  } else if (Backend != backend::opencl) {
    // OpenCL backend support is conditional on the cl_khr_command_buffer
    // extension being available.
    assert(!SupportsGraphs);
  }
}
