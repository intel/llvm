// RUN: %{build} -o %t.out
// RUN: %{run} %t.out

// Concurrent access to shared USM allocations is not supported by CUDA on
// Windows, this occurs when the host-task and device kernel both access
// USM without a dependency between the commands.
// UNSUPPORTED: cuda && windows

// REQUIRES: aspect-usm_shared_allocations

#include <sycl/detail/core.hpp>
#include <sycl/usm.hpp>

int main() {
  using namespace sycl;
  queue Queue{};

  constexpr size_t Size = 1024;
  int *PtrA = malloc_shared<int>(Size, Queue);
  int *PtrB = malloc_shared<int>(Size, Queue);

  Queue.submit([&](handler &CGH) {
    CGH.parallel_for(range<1>(Size), [=](item<1> id) { PtrA[id] = id; });
  });

  const int ConstValue = 42;
  Queue.submit([&](handler &CGH) {
    CGH.host_task([=]() {
      for (size_t i = 0; i < Size; i++) {
        PtrB[i] = ConstValue;
      }
    });
  });

  Queue.wait_and_throw();

  for (size_t i = 0; i < Size; i++) {
    assert(i == PtrA[i]);
    assert(ConstValue == PtrB[i]);
  }

  free(PtrA, Queue);
  free(PtrB, Queue);

  return 0;
}
