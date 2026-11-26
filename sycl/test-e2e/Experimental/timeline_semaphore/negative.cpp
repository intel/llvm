// REQUIRES: aspect-ext_oneapi_timeline_semaphore
// RUN: %{build} -o %t.out
// RUN: %{run} %t.out

// Test: the enqueue functions throw errc::invalid when the queue and the
// semaphore have different contexts, and when the queue is recording a graph.

#include <sycl/detail/core.hpp>
#include <sycl/ext/oneapi/experimental/graph.hpp>
#include <sycl/ext/oneapi/experimental/timeline_semaphore.hpp>

#include <cassert>
#include <functional>

namespace syclex = sycl::ext::oneapi::experimental;

static bool throwsInvalid(const std::function<void()> &F) {
  try {
    F();
  } catch (const sycl::exception &E) {
    return E.code() == sycl::errc::invalid;
  }
  return false;
}

int main() {
  sycl::queue q;

  sycl::context other_ctxt{q.get_device()};
  syclex::timeline_semaphore other_sem{other_ctxt};
  assert(throwsInvalid([&] { syclex::enqueue_signal(q, other_sem, 1); }));
  assert(throwsInvalid([&] { syclex::enqueue_wait(q, other_sem, 1); }));

  if (q.get_device().has(sycl::aspect::ext_oneapi_limited_graph)) {
    syclex::timeline_semaphore sem{q.get_context()};
    syclex::command_graph graph{q.get_context(), q.get_device()};
    graph.begin_recording(q);
    assert(throwsInvalid([&] { syclex::enqueue_signal(q, sem, 1); }));
    assert(throwsInvalid([&] { syclex::enqueue_wait(q, sem, 1); }));
    graph.end_recording(q);
  }

  return 0;
}
