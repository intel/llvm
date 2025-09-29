// REQUIRES: aspect-usm_shared_allocations
// RUN: %{build} -o %t.out
// RUN: %{run} %t.out

// This test checks whether the free function command mem_advise operation is
// valid.
#define __DPCPP_ENABLE_UNFINISHED_KHR_EXTENSIONS

#include <sycl/detail/core.hpp>
#include <sycl/khr/free_function_commands.hpp>
#include <sycl/usm.hpp>

int main() {

  int Failed = 0;
  constexpr size_t N = 1024;

  sycl::context Context;
  sycl::queue Queue;

  int *Memory = sycl::malloc_shared<int>(N, Queue);

  constexpr size_t ChunkSize = N / 3;
  try {
    sycl::khr::mem_advise(Queue, Memory, ChunkSize, 0);
  } catch (sycl::exception &Excep) {
    std::cout << "SYCL exception caught:" << Excep.what() << std::endl;
    ++Failed;
  }

  try {
    sycl::khr::mem_advise(Queue, Memory, ChunkSize, 0);
  } catch (sycl::exception &Excep) {
    std::cout << "SYCL exception caught:" << Excep.what() << std::endl;
    ++Failed;
  }

  try {
    sycl::khr::submit(Queue, [&](sycl::handler &CGH) {
      sycl::khr::mem_advise(CGH, Memory + ChunkSize, ChunkSize, 0);
    });
  } catch (sycl::exception &Excep) {
    std::cout << "SYCL exception caught:" << Excep.what() << std::endl;
    ++Failed;
  }

  try {
    sycl::event Event =
        sycl::khr::submit_tracked(Queue, [&](sycl::handler &Handler) {
          sycl::khr::mem_advise(Handler, Memory + ChunkSize * 2, ChunkSize, 0);
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
