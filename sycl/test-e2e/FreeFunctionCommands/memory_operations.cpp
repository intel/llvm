// REQUIRES: aspect-usm_shared_allocations
// RUN: %{build} -o %t.out
// RUN: %{run} %t.out

// This test checks whether the free function command memory operations are
// valid.
#define __DPCPP_ENABLE_UNFINISHED_KHR_EXTENSIONS

#include <sycl/detail/core.hpp>
#include <sycl/khr/free_function_commands.hpp>
#include <sycl/usm.hpp>

#include "helpers.hpp"

int main() {
  int Failed = 0;
  sycl::queue Queue;
  constexpr size_t N = 1024;

  {
    int *Memory1 = sycl::malloc_shared<int>(N, Queue);
    int *Memory2 = sycl::malloc_shared<int>(N, Queue);
    std::fill(Memory1, Memory1 + N, 20);
    std::fill(Memory2, Memory2 + N, 21);

    sycl::khr::submit(Queue, [&](sycl::handler &Handler) {
      sycl::khr::fill(Handler, Memory1, 999, N);
    });
    Queue.wait();
    for (size_t i = 0; i < N; ++i)
      Failed += Check(Memory1, 999, i, "fill_with_submit");

    sycl::khr::submit(Queue, [&](sycl::handler &Handler) {
      sycl::khr::memcpy(Handler, Memory2, Memory1, N * sizeof(int));
    });
    Queue.wait();
    for (size_t i = 0; i < N; ++i)
      Failed += Check(Memory2, Memory1[i], i, "memcpy_with_submit");

    sycl::khr::submit(Queue, [&](sycl::handler &Handler) {
      sycl::khr::memset(Handler, Memory2, 0, N * sizeof(int));
    });
    Queue.wait();
    for (size_t i = 0; i < N; ++i)
      Failed += Check(Memory2, 0, i, "memset_with_submit");

    sycl::khr::submit(Queue, [&](sycl::handler &Handler) {
      sycl::khr::copy(Handler, Memory1, Memory2, N);
    });
    Queue.wait();
    for (size_t i = 0; i < N; ++i)
      Failed += Check(Memory2, Memory1[i], i, "copy_with_submit");

    sycl::free(Memory1, Queue);
    sycl::free(Memory2, Queue);
  }

  {
    int *Memory1 = sycl::malloc_shared<int>(N, Queue);
    int *Memory2 = sycl::malloc_shared<int>(N, Queue);
    std::fill(Memory1, Memory1 + N, 40);
    std::fill(Memory2, Memory2 + N, 41);

    sycl::khr::submit_tracked(Queue, [&](sycl::handler &Handler) {
      sycl::khr::fill(Handler, Memory1, 777, N);
    }).wait();

    for (size_t i = 0; i < N; ++i)
      Failed += Check(Memory1, 777, i, "fill_with_submit_tracked");

    sycl::khr::submit_tracked(Queue, [&](sycl::handler &Handler) {
      sycl::khr::memcpy(Handler, Memory2, Memory1, N * sizeof(int));
    }).wait();

    for (size_t i = 0; i < N; ++i)
      Failed += Check(Memory2, Memory1[i], i, "memcpy_with_submit_tracked");

    sycl::khr::submit_tracked(Queue, [&](sycl::handler &Handler) {
      sycl::khr::memset(Handler, Memory2, 0, N * sizeof(int));
    }).wait();

    for (size_t i = 0; i < N; ++i)
      Failed += Check(Memory2, 0, i, "memset_with_submit_tracked");

    sycl::khr::submit_tracked(Queue, [&](sycl::handler &Handler) {
      sycl::khr::copy(Handler, Memory1, Memory2, N);
    }).wait();

    for (size_t i = 0; i < N; ++i)
      Failed += Check(Memory2, Memory1[i], i, "copy_with_submit_tracked");

    sycl::free(Memory1, Queue);
    sycl::free(Memory2, Queue);
  }

  {
    int *Memory1 = sycl::malloc_shared<int>(N, Queue);
    int *Memory2 = sycl::malloc_shared<int>(N, Queue);
    std::fill(Memory1, Memory1 + N, 60);
    std::fill(Memory2, Memory2 + N, 61);

    sycl::khr::fill(Queue, Memory1, 333, N);
    Queue.wait();
    for (size_t i = 0; i < N; ++i)
      Failed += Check(Memory1, 333, i, "fill_shortcut");

    sycl::khr::memcpy(Queue, Memory2, Memory1, N * sizeof(int));
    Queue.wait();
    for (size_t i = 0; i < N; ++i)
      Failed += Check(Memory2, Memory1[i], i, "memcpy_shortcut");

    sycl::khr::memset(Queue, Memory2, 0, N * sizeof(int));
    Queue.wait();
    for (size_t i = 0; i < N; ++i)
      Failed += Check(Memory2, 0, i, "memset_shortcut");

    sycl::khr::copy(Queue, Memory1, Memory2, N);
    Queue.wait();
    for (size_t i = 0; i < N; ++i)
      Failed += Check(Memory2, Memory1[i], i, "copy_shortcut");

    sycl::free(Memory1, Queue);
    sycl::free(Memory2, Queue);
  }

  return Failed;
}
