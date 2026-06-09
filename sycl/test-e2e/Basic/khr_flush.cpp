// RUN: %{build} -o %t.out
// RUN: %{run} %t.out

// Test checks that queue::khr_flush() issues commands in the queue to the
// device.

#include <sycl/detail/core.hpp>
#include <sycl/usm.hpp>

static void CheckArray(int *x, size_t buffer_size, int expected) {
  for (size_t i = 0; i < buffer_size; ++i) {
    assert(x[i] == expected);
  }
}

using namespace sycl;

void TestFunc(queue &Q) {
  static constexpr int Size = 100;

  int *X = malloc_host<int>(Size, Q);
  int *Y = malloc_host<int>(Size, Q);

  // First, check with single_task()
  auto FillEv = Q.fill(X, 99, Size);
  auto SingleTaskEv = Q.submit([&](handler &CGH) {
    auto SingleTask = [=] {
      for (int I = 0; I < Size; I++)
        X[I] += 1;
    };
    CGH.single_task(SingleTask);
  });

  // Call khr_flush() to flush the commands and wait for them to complete
  Q.khr_flush();
  while (SingleTaskEv.get_info<sycl::info::event::command_execution_status>() !=
         sycl::info::event_command_status::complete) {
  };

  // Check that the commands are indeed complete
  CheckArray(X, Size, 100);

  // Now check again with parallel_for()
  auto MemCpyEv = Q.copy(X, Y, Size);
  auto ParallelTaskEv = Q.submit([&](handler &CGH) {
    CGH.parallel_for<class Kernel1>(sycl::range<1>(Size),
                                    [=](sycl::id<1> WI) { Y[WI] *= 2; });
  });

  Q.khr_flush();
  while (
      ParallelTaskEv.get_info<sycl::info::event::command_execution_status>() !=
      sycl::info::event_command_status::complete) {
  };

  CheckArray(Y, Size, 200);

  free(X, Q);
  free(Y, Q);
}

int main() {
  queue Q1{};
  TestFunc(Q1);

  return 0;
}
