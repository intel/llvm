// REQUIRES: aspect-usm_shared_allocations
// RUN: %{build} -o %t.out
// RUN: %{run} %t.out

// Tests the enqueue free function memory operations.

#include <sycl/detail/core.hpp>
#include <sycl/ext/oneapi/experimental/enqueue_functions.hpp>
#include <sycl/usm.hpp>

#include "common.hpp"

#include <algorithm>

namespace oneapiext = sycl::ext::oneapi::experimental;

constexpr size_t N = 1024;

int main() {
  sycl::queue Q;

  int Failed = 0;

  // Shortcuts.
  {
    int *Memory1 = sycl::malloc_shared<int>(N, Q);
    int *Memory2 = sycl::malloc_shared<int>(N, Q);
    std::fill(Memory1, Memory1 + N, 42);
    std::fill(Memory2, Memory2 + N, 24);

    oneapiext::fill(Q, Memory1, 1, N - 20);
    Q.wait();
    for (size_t I = 0; I < N; ++I)
      Failed += Check(Memory1, (I < N - 20 ? 1 : 42), I, "fill shortcut");

    oneapiext::memcpy(Q, Memory2, Memory1, N * sizeof(int));
    Q.wait();
    for (size_t I = 0; I < N; ++I)
      Failed += Check(Memory2, Memory1[I], I, "memcpy shortcut");

    oneapiext::memset(Q, Memory2, 0, (N - 10) * sizeof(int));
    Q.wait();
    for (size_t I = 0; I < N; ++I)
      Failed += Check(Memory2, (I < N - 10 ? 0 : 42), I, "memset shortcut");

    oneapiext::copy(Q, Memory1, Memory2, N);
    Q.wait();
    for (size_t I = 0; I < N; ++I)
      Failed += Check(Memory2, Memory1[I], I, "copy shortcut");

    sycl::free(Memory1, Q);
    sycl::free(Memory2, Q);
  }

  // Submit without event.
  {
    int *Memory1 = sycl::malloc_shared<int>(N, Q);
    int *Memory2 = sycl::malloc_shared<int>(N, Q);
    std::fill(Memory1, Memory1 + N, 42);
    std::fill(Memory2, Memory2 + N, 24);

    oneapiext::submit(Q, [&](sycl::handler &CGH) {
      oneapiext::fill(CGH, Memory1, 1, N - 20);
    });
    Q.wait();
    for (size_t I = 0; I < N; ++I)
      Failed += Check(Memory1, (I < N - 20 ? 1 : 42), I, "fill without event");

    oneapiext::submit(Q, [&](sycl::handler &CGH) {
      oneapiext::memcpy(CGH, Memory2, Memory1, N * sizeof(int));
    });
    Q.wait();
    for (size_t I = 0; I < N; ++I)
      Failed += Check(Memory2, Memory1[I], I, "memcpy without event");

    oneapiext::submit(Q, [&](sycl::handler &CGH) {
      oneapiext::memset(CGH, Memory2, 0, (N - 10) * sizeof(int));
    });
    Q.wait();
    for (size_t I = 0; I < N; ++I)
      Failed +=
          Check(Memory2, (I < N - 10 ? 0 : 42), I, "memset without event");

    oneapiext::submit(Q, [&](sycl::handler &CGH) {
      oneapiext::copy(CGH, Memory1, Memory2, N);
    });
    Q.wait();
    for (size_t I = 0; I < N; ++I)
      Failed += Check(Memory2, Memory1[I], I, "copy without event");

    sycl::free(Memory1, Q);
    sycl::free(Memory2, Q);
  }

  // Submit with event.
  {
    int *Memory1 = sycl::malloc_shared<int>(N, Q);
    int *Memory2 = sycl::malloc_shared<int>(N, Q);
    std::fill(Memory1, Memory1 + N, 42);
    std::fill(Memory2, Memory2 + N, 24);

    oneapiext::submit_with_event(Q, [&](sycl::handler &CGH) {
      oneapiext::fill(CGH, Memory1, 1, N - 20);
    }).wait();
    for (size_t I = 0; I < N; ++I)
      Failed += Check(Memory1, (I < N - 20 ? 1 : 42), I, "fill with event");

    oneapiext::submit_with_event(Q, [&](sycl::handler &CGH) {
      oneapiext::memcpy(CGH, Memory2, Memory1, N * sizeof(int));
    }).wait();
    for (size_t I = 0; I < N; ++I)
      Failed += Check(Memory2, Memory1[I], I, "memcpy with event");

    oneapiext::submit_with_event(Q, [&](sycl::handler &CGH) {
      oneapiext::memset(CGH, Memory2, 0, (N - 10) * sizeof(int));
    }).wait();
    for (size_t I = 0; I < N; ++I)
      Failed += Check(Memory2, (I < N - 10 ? 0 : 42), I, "memset with event");

    oneapiext::submit_with_event(Q, [&](sycl::handler &CGH) {
      oneapiext::copy(CGH, Memory1, Memory2, N);
    }).wait();
    for (size_t I = 0; I < N; ++I)
      Failed += Check(Memory2, Memory1[I], I, "copy with event");

    sycl::free(Memory1, Q);
    sycl::free(Memory2, Q);
  }

  return Failed;
}
