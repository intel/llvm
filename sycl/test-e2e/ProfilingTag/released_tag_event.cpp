// REQUIRES: aspect-ext_oneapi_queue_profiling_tag
// RUN: %{build} -o %t.out
// RUN: %{run} %t.out

// Releasing the event of a profiling tag that is still in flight must not
// affect the tags submitted after it.

#include <sycl/detail/core.hpp>
#include <sycl/ext/oneapi/experimental/profiling_tag.hpp>

#include <optional>
#include <vector>

namespace syclex = sycl::ext::oneapi::experimental;

constexpr size_t Iterations = 1000;
// Several tags in flight per synchronization, so a released one is up for
// reuse by the next one.
constexpr size_t TagsPerIteration = 10;

int main() {
  sycl::queue Queue{sycl::property::queue::in_order()};

  for (size_t I = 0; I < Iterations; ++I) {
    std::vector<sycl::event> Tags;
    Tags.reserve(TagsPerIteration);

    for (size_t J = 0; J < TagsPerIteration; ++J) {
      // Drop this tag's event while its recording is still in flight, with
      // nothing else keeping it alive.
      {
        std::optional<sycl::event> Dropped{syclex::submit_profiling_tag(Queue)};
        Dropped.reset();
      }

      // The pool may hand the released event back for this tag, but only once
      // its previous write is done - the timestamp below must be this tag's.
      Tags.push_back(syclex::submit_profiling_tag(Queue));
    }

    Queue.wait_and_throw();

    // A tag that lost its timestamp reports zero, or throws
    // UR_RESULT_ERROR_PROFILING_INFO_NOT_AVAILABLE out of main.
    for (sycl::event &Tag : Tags)
      if (Tag.get_profiling_info<sycl::info::event_profiling::command_end>() ==
          0)
        return 1;
  }

  return 0;
}
