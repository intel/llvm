// REQUIRES: aspect-usm_shared_allocations
// RUN: %{build} -o %t.out
// RUN: %{run} %t.out

// This test checks whether the free function command memory operations are
// valid.

#include <sycl/detail/core.hpp>
#include <sycl/ext/khr/free_function_commands.hpp>
#include <sycl/usm.hpp>

#include "helpers.hpp"

int main() {
  int Failed = 0;
  sycl::queue Queue;
  constexpr size_t N = 1024;

  int *Memory1 = sycl::malloc_shared<int>(N, Queue);
  int *Memory2 = sycl::malloc_shared<int>(N, Queue);
  std::fill(Memory1, Memory1 + N, 20);
  std::fill(Memory2, Memory2 + N, 21);

  sycl::khr::submit(Queue, [&](sycl::handler &Handler) {
    sycl::khr::fill(Handler, Memory1, 999, N);
  });
  Queue.wait();
  for (size_t i = 0; i < N; ++i)
    Failed += Check(Memory1, 999, i, "fill");

  sycl::khr::submit(Queue, [&](sycl::handler &Handler) {
    sycl::khr::memcpy(Handler, Memory2, Memory1, N * sizeof(int));
  });
  Queue.wait();
  for (size_t i = 0; i < N; ++i)
    Failed += Check(Memory2, Memory1[i], i, "memcpy");

  sycl::khr::submit(Queue, [&](sycl::handler &Handler) {
    sycl::khr::memset(Handler, Memory2, 0, N * sizeof(int));
  });
  Queue.wait();
  for (size_t i = 0; i < N; ++i)
    Failed += Check(Memory2, 0, i, "memset");

  sycl::khr::submit(Queue, [&](sycl::handler &Handler) {
    sycl::khr::copy(Handler, Memory1, Memory2, N);
  });
  Queue.wait();
  for (size_t i = 0; i < N; ++i)
    Failed += Check(Memory2, Memory1[i], i, "copy");

  sycl::free(Memory1, Queue);
  sycl::free(Memory2, Queue);

  return Failed;
}