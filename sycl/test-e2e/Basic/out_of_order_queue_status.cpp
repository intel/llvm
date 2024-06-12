// RUN: %{build} -o %t.out
// RUN: %{run} %t.out

// Test checks that queue::ext_oneapi_empty() returns status of the out-of-order
// queue.

#include <sycl/detail/core.hpp>
#include <sycl/detail/host_task_impl.hpp>
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

  assert(Q.ext_oneapi_empty() && "Queue is expected to be empty");

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

  // Wait a bit to give a chance for tasks to complete.
  std::this_thread::sleep_for(std::chrono::milliseconds(500));

  // We expect that all submitted tasks are finished if ext_oneapi_empty is
  // true.
  if (Q.ext_oneapi_empty())
    CheckArray(Y, Size, 200);

  Q.wait();

  // After synchronization queue must be empty.
  assert(Q.ext_oneapi_empty() && "Queue is expected to be empty");

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

  // Feature is not supported for OpenCL, exception must be thrown.
  if (Q.get_device().get_backend() == backend::opencl)
    return ExceptionThrown ? 0 : -1;

  return 0;
}
