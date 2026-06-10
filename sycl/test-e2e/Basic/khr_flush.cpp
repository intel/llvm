// RUN: %{build} -o %t.out
// RUN: %{run} %t.out

// Test checks that queue::khr_flush() issues commands in the queue to the
// device.

#include <sycl/detail/core.hpp>
#include <sycl/properties/all_properties.hpp>
#include <sycl/usm.hpp>

#include <chrono>
#include <thread>

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
  std::this_thread::sleep_for(std::chrono::milliseconds(500));
  auto Deadline = std::chrono::steady_clock::now() + std::chrono::seconds(30);
  while (SingleTaskEv.get_info<sycl::info::event::command_execution_status>() !=
         sycl::info::event_command_status::complete) {
    if (std::chrono::steady_clock::now() > Deadline) {
      assert(false &&
             "single_task in khr_flush test did not complete within 30s");
      break;
    }
    std::this_thread::sleep_for(std::chrono::milliseconds(10));
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
  std::this_thread::sleep_for(std::chrono::milliseconds(500));
  Deadline = std::chrono::steady_clock::now() + std::chrono::seconds(60);
  while (
      ParallelTaskEv.get_info<sycl::info::event::command_execution_status>() !=
      sycl::info::event_command_status::complete) {
    if (std::chrono::steady_clock::now() > Deadline) {
      assert(false &&
             "parallel_for in khr_flush test did not complete within 60s");
      break;
    }
    std::this_thread::sleep_for(std::chrono::milliseconds(10));
  };

  CheckArray(Y, Size, 200);

  free(X, Q);
  free(Y, Q);
}

int main() {
  queue Q1{property::queue::in_order()};
  TestFunc(Q1);

  return 0;
}
