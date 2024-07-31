// REQUIRES: aspect-ext_oneapi_queue_profiling_tag
// RUN: %{build} -o %t.out
// RUN: %{run} %t.out

// Tests the get_profiling_tag extension function on an out-of-order queue.

// HIP backend currently returns invalid values for submission time queries.
// https://github.com/intel/llvm/issues/12904
// UNSUPPORTED: hip

// CUDA backend seems to fail sporadically for expected profiling tag time
// query orderings.
// https://github.com/intel/llvm/issues/14053
// UNSUPPORTED: cuda

#include "common.hpp"

int main() {
  sycl::queue Queue;
  return run_test_on_queue(Queue);
}
