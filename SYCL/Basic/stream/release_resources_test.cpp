// RUN: %clangxx -fsycl -fsycl-targets=%sycl_triple %s -o %t.out
// RUN: env SYCL_PI_TRACE=2 %CPU_RUN_PLACEHOLDER %t.out %CPU_CHECK_PLACEHOLDER
// RUN: env SYCL_PI_TRACE=2 %GPU_RUN_PLACEHOLDER %t.out %GPU_CHECK_PLACEHOLDER
// RUN: env SYCL_PI_TRACE=2 %ACC_RUN_PLACEHOLDER %t.out %ACC_CHECK_PLACEHOLDER

// Check that buffer used by a stream object is released.

#include <sycl/sycl.hpp>

using namespace cl::sycl;

int main() {
  {
    queue Queue;

    // CHECK:---> piMemRelease
    Queue.submit([&](handler &CGH) {
      stream Out(1024, 80, CGH);
      CGH.parallel_for<class test_cleanup1>(
          range<1>(2), [=](id<1> i) { Out << "Hello, World!" << endl; });
    });
    Queue.wait();
  }

  return 0;
}
