// REQUIRES: aspect-usm_shared_allocations
// RUN: %{build} -o %t.out
// RUN: %{run} %t.out

// This test checks whether the free function command launch task is valid.

#include <sycl/detail/core.hpp>
#include <sycl/ext/khr/free_function_commands.hpp>
#include <sycl/usm.hpp>

#include <array>

#include "helpers.hpp"

int main() {

  constexpr size_t N = 1024;
  int Failed = 0;

  sycl::queue Queue;

  {
    std::array<int, N> DataBuffer = {-0};

    {
      sycl::buffer<int, 1> MemBuffer{DataBuffer.data(), sycl::range<1>{N}};
      sycl::khr::submit(Queue, [&](sycl::handler &Handler) {
        sycl::accessor MemAcc{MemBuffer, Handler, sycl::write_only};
        sycl::khr::launch_task(Handler, [=]() {
          for (size_t i = 0; i < N; ++i) {
            MemAcc[i] = 101;
          }
        });
      });
    }
    for (size_t i = 0; i < N; ++i)
      Failed += Check(DataBuffer.data(), 101, i, "launch_task_with_buffer");
  }

  {
    int *DataBuffer = sycl::malloc_shared<int>(N, Queue);

    sycl::khr::launch_task(Queue, [=]() {
      for (size_t i = 0; i < N; ++i) {
        DataBuffer[i] = 102;
      }
    });

    Queue.wait();

    for (size_t i = 0; i < N; ++i)
      Failed += Check(DataBuffer, 102, i, "launch_task_with_usm");

    sycl::free(DataBuffer, Queue);
  }

  return Failed;
}
