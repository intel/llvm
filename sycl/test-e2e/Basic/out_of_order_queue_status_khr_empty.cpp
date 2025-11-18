// RUN: %{build} -o %t.out
// RUN: %{run} %t.out

// Test checks that queue::khr_empty() returns status of the out-of-order
// queue.

#define __DPCPP_ENABLE_UNFINISHED_KHR_EXTENSIONS

#include <sycl/detail/core.hpp>
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

  assert(Q.khr_empty() && "Queue is expected to be empty");

  int *X = malloc_host<int>(Size, Q);
  int *Y = malloc_host<int>(Size, Q);

  auto FillEv = Q.fill(X, 99, Size);
  auto HostEv = Q.submit([&](handler &CGH) {
    CGH.depends_on(FillEv);
    auto HostTask = [=] {
      for (int I = 0; I < Size; I++)
        X[I] += 1;
    };
    CGH.host_task(HostTask);
  });
  auto MemCpyEv = Q.copy(X, Y, Size, {HostEv});
  constexpr int NumIter = 5;
  for (int I = 0; I < NumIter; I++) {
    Q.submit([&](handler &CGH) {
      CGH.depends_on(MemCpyEv);
      CGH.parallel_for<class Kernel1>(
          sycl::range<1>(Size / NumIter),
          [=](sycl::id<1> WI) { Y[WI + I * Size / NumIter] *= 2; });
    });
  }

  constexpr int InfiniteLoopPreventionThreshold = 100;
  int InfiniteLoopPreventionCounter = 0;

  // Wait for tasks created by parallel_for to actually start running. Otherwise
  // khr_empty may return true not because the queue is completed, but because
  // tasks haven't been added to the queue.
  do {
    std::this_thread::sleep_for(std::chrono::milliseconds(300));
    InfiniteLoopPreventionCounter++;
  } while (Y[0] == 100 &&
           InfiniteLoopPreventionCounter < InfiniteLoopPreventionThreshold);

  assert(InfiniteLoopPreventionCounter < InfiniteLoopPreventionThreshold &&
         "Failure since test took too long to run");

  // Wait a bit to give a chance for tasks to complete.
  std::this_thread::sleep_for(std::chrono::milliseconds(1000));

  // We expect that all submitted tasks are finished if khr_empty is true.
  if (Q.khr_empty() && Y[0] != 100)
    CheckArray(Y, Size, 200);

  Q.wait();

  // After synchronization queue must be empty.
  assert(Q.khr_empty() && "Queue is expected to be empty");

  free(X, Q);
  free(Y, Q);
}

int main() {
  queue Q;

  bool ExceptionThrown = false;
  try {
    TestFunc(Q);
  } catch (sycl::exception &E) {
    ExceptionThrown = true;
  }

  return ExceptionThrown ? -1 : 0;
}
