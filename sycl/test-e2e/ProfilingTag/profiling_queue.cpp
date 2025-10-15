// REQUIRES: aspect-queue_profiling

// RUN: %{build} -o %t.out
// RUN: %{run} %t.out

// Tests the get_profiling_tag extension function on an out-of-order queue with
// profiling enabled.
// Note: Extension should work even on devices that do not support the
//       ext_oneapi_queue_profiling_tag aspect.

// Bug in OpenCL GPU driver causes fallback solution to return end time later
// than the submission of the following work.
// UNSUPPORTED: opencl && gpu

// HIP backend currently returns invalid values for submission time queries.
// UNSUPPORTED: hip
// UNSUPPORTED-TRACKER: https://github.com/intel/llvm/issues/12904

// Flaky on CUDA
// UNSUPPORTED: cuda
// UNSUPPORTED-TRACKER: https://github.com/intel/llvm/issues/14053

// UNSUPPORTED: level_zero_v2_adapter
// UNSUPPORTED-TRACKER: https://github.com/intel/llvm/issues/19116
//
// UNSUPPORTED: ((intel_gpu_acm{{.*}} || intel_gpu_pvc || intel_gpu_bmg{{.*}}) && (!level_zero)) || preview-mode
// UNSUPPORTED-TRACKER: https://github.com/intel/llvm/issues/19425

// XFAIL: target-native_cpu
// XFAIL-TRACKER: https://github.com/intel/llvm/issues/20142

#include "common.hpp"

int main() {
  sycl::property_list Properties{sycl::property::queue::enable_profiling()};
  sycl::queue Queue{Properties};
  return run_test_on_queue(Queue);
}
