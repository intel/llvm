// REQUIRES: aspect-usm_shared_allocations
// RUN: %{build} -o %t.out
// RUN: %{run} %t.out

// This test checks whether the free function command prefetch operation is
// valid.
#define __DPCPP_ENABLE_UNFINISHED_KHR_EXTENSIONS

#include <sycl/detail/core.hpp>
#include <sycl/khr/free_function_commands.hpp>
#include <sycl/usm.hpp>

int main() {
  int Failed = 0;
  constexpr size_t N = 1024;
  constexpr size_t ChunkSize = N / 3;

  sycl::queue Queue;

  int *Memory = sycl::malloc_shared<int>(N, Queue);
  try {
    sycl::khr::prefetch(Queue, Memory, ChunkSize);
  } catch (sycl::exception &Excep) {
    std::cout << "SYCL exception caught:" << Excep.what() << std::endl;
    ++Failed;
  }

  try {
    sycl::khr::submit(Queue, [&](sycl::handler &Handler) {
      sycl::khr::prefetch(Handler, Memory + ChunkSize, ChunkSize);
    });
  } catch (sycl::exception &Excep) {
    std::cout << "SYCL exception caught:" << Excep.what() << std::endl;
    ++Failed;
  }

  try {
    sycl::event Event =
        sycl::khr::submit_tracked(Queue, [&](sycl::handler &Handler) {
          sycl::khr::prefetch(Handler, Memory + ChunkSize * 2, ChunkSize);
        });
    Event.wait();
  } catch (sycl::exception &Excep) {
    std::cout << "SYCL exception caught:" << Excep.what() << std::endl;
    ++Failed;
  }

  Queue.wait();
  sycl::free(Memory, Queue);

  return Failed;
}
