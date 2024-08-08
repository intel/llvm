// REQUIRES: level_zero

// RUN: %{build} -o %t.out
// RUN: %{l0_leak_check} %{run} %t.out 2>&1 | FileCheck %s --implicit-check-not=LEAK

// Regression test to avoid the reintroduction of a leak in L0 in the profiling
// tags when using barriers to ensure ordering on out-of-order queues.

#include <sycl/detail/core.hpp>
#include <sycl/ext/oneapi/experimental/profiling_tag.hpp>

int main() {
  sycl::queue Queue;
  sycl::event TagE =
      sycl::ext::oneapi::experimental::submit_profiling_tag(Queue);
  Queue.wait();
  return 0;
}
