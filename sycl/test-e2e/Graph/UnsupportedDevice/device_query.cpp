// RUN: %{build} -o %t.out
// RUN: %{run} %t.out

// OpenCL support depends on extensions
// UNSUPPORTED: opencl

// Tests the using device query for graphs support, and that the return value
// matches expectations.

#include "../graph_common.hpp"

int main() {
  queue Queue;

  auto Device = Queue.get_device();
  bool SupportsGraphs = Device.has(aspect::ext_oneapi_graph);
  bool SupportsLimitedGraphs = Device.has(aspect::ext_oneapi_limited_graph);
  auto Backend = Device.get_backend();

  if (Backend == backend::ext_oneapi_level_zero) {
    // Full graph support is dependent on the Level Zero device & driver,
    // and cannot be asserted without diving into these details.
    assert(SupportsLimitedGraphs);
  } else if ((Backend == backend::ext_oneapi_cuda) ||
             (Backend == backend::ext_oneapi_hip)) {
    assert(SupportsGraphs);
    assert(SupportsLimitedGraphs);
  } else {
    assert(!SupportsGraphs);
    assert(!SupportsLimitedGraphs);
  }
}
