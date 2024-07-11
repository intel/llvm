// UNSUPPORTED: windows || hip_amd
// RUN: %{build} -o %t.out
// RUN: not env SYCL_TRACE_TERMINATE_ON_WARNING=1 %{run} sycl-trace --verify %t.out | FileCheck %s

// Test parameter analysis of USM usage

#include <sycl/detail/core.hpp>
#include <sycl/ext/oneapi/experimental/builtins.hpp>

int main() {
  sycl::queue Q;
  unsigned int *AllocSrc = nullptr;
  try {
    // CHECK: [USM] Function uses nullptr as kernel parameter with index = 0.
    // CHECK: | kernel location: function main at {{.*}}queue_single_task_nullptr.cpp:[[# @LINE + 1 ]]
    Q.single_task([=]() {
      if (AllocSrc == nullptr)
        sycl::ext::oneapi::experimental::printf("nullptr");
    });
    Q.wait();
  } catch (...) {
  }
  return 0;
}
