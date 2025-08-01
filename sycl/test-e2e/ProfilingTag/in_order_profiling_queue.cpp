// REQUIRES: aspect-queue_profiling

// RUN: %{build} -o %t.out
// RUN: %{run} %t.out

// Tests the get_profiling_tag extension function on an in-order queue with
// profiling enabled.
// Note: Extension should work even on devices that do not support the
//       ext_oneapi_queue_profiling_tag aspect.

// Bug in OpenCL GPU driver causes fallback solution to return end time later
// than the submission of the following work.
// UNSUPPORTED: opencl && gpu

// HIP backend currently returns invalid values for submission time queries.
// UNSUPPORTED: hip
// UNSUPPORTED-TRACKER: https://github.com/intel/llvm/issues/12904

// CUDA backend seems to fail sporadically for expected profiling tag time
// query orderings.
// UNSUPPORTED: cuda
// UNSUPPORTED-TRACKER: https://github.com/intel/llvm/issues/14053

#include "common.hpp"

int main() {
  sycl::property_list Properties{sycl::property::queue::in_order(),
                                 sycl::property::queue::enable_profiling()};
  sycl::queue Queue{Properties};
  return run_test_on_queue(Queue);
}
