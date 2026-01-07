// REQUIRES: aspect-ext_oneapi_queue_profiling_tag
// RUN: %{build} -o %t.out
// RUN: %{run} %t.out

// Regression test to ensure that the adapters do not tamper (e.g. through
// caching the events) the recordings.

// HIP backend currently returns invalid values for submission time queries.
// UNSUPPORTED: hip
// UNSUPPORTED-TRACKER: https://github.com/intel/llvm/issues/12904

// CUDA backend seems to fail sporadically for expected profiling tag time
// query orderings.
// UNSUPPORTED: cuda
// UNSUPPORTED-TRACKER: https://github.com/intel/llvm/issues/14053

#include <sycl/detail/core.hpp>
#include <sycl/ext/oneapi/experimental/profiling_tag.hpp>

int main() {
  sycl::queue Q{sycl::property::queue::in_order()};

  uint64_t T1 = 0;
  for (size_t I = 0; I < 20; ++I) {
    sycl::event E = sycl::ext::oneapi::experimental::submit_profiling_tag(Q);
    uint64_t T2 =
        E.get_profiling_info<sycl::info::event_profiling::command_end>();
    assert(T1 < T2);
    std::swap(T1, T2);
  }

  return 0;
}
