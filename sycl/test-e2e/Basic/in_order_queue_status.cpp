// RUN: %clangxx -fsycl -fsycl-targets=%sycl_triple %s -o %t.out
// RUN: %ACC_RUN_PLACEHOLDER %t.out
// RUN: %CPU_RUN_PLACEHOLDER %t.out
// RUN: %GPU_RUN_PLACEHOLDER %t.out

// Test checks that queue::ext_oneapi_empty() returns status of the in-order
// queue.

#include <chrono>
#include <sycl.hpp>
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
  auto SingleTaskEv = Q.submit([&](handler &CGH) {
    auto SingleTask = [=] {
      for (int I = 0; I < Size; I++)
        X[I] += 1;
    };
    CGH.single_task(SingleTask);
  });
  auto MemCpyEv = Q.copy(X, Y, Size);
  constexpr int NumIter = 5;
  for (int I = 0; I < NumIter; I++) {
    Q.submit([&](handler &CGH) {
      CGH.parallel_for<class Kernel1>(sycl::range<1>(Size),
                                      [=](sycl::id<1> WI) { Y[WI] *= 2; });
    });
  }

  // Wait a bit to give a chance for tasks to complete.
  std::this_thread::sleep_for(std::chrono::milliseconds(500));

  // We expect that all submitted tasks are finished if ext_oneapi_empty is
  // true.
  if (Q.ext_oneapi_empty())
    CheckArray(Y, Size, 3200);

  Q.wait();

  // After synchronization queue must be empty.
  assert(Q.ext_oneapi_empty() && "Queue is expected to be empty");

  free(X, Q);
  free(Y, Q);
}

int main() {
  // Test in-order queue.
  queue Q1{property::queue::in_order()};
  TestFunc(Q1);

  // Test in-order queue with discard_events property.
  sycl::property_list Props{
      property::queue::in_order{},
      sycl::ext::oneapi::property::queue::discard_events{}};
  queue Q2{Props};

  bool ExceptionThrown = false;
  try {
    TestFunc(Q2);
  } catch (sycl::exception &E) {
    ExceptionThrown = true;
  }

  // Feature is not supported for OpenCL, exception must be thrown.
  if (Q2.get_device().get_backend() == backend::opencl)
    return ExceptionThrown ? 0 : -1;

  return 0;
}
