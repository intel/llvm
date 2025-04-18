// UNSUPPORTED: windows
// UNSUPPORTED-INTENDED: We can't safely release any resoureces on Windows, so
// the test is expected to fail there. See comments in
// GlobalHandler::releaseDefaultContexts.
// RUN: %{build} -o %t.out
// RUN: env SYCL_UR_TRACE=2 %{run} %t.out | FileCheck %s

// Check that buffer used by a stream object is released.

#include <sycl/detail/core.hpp>

#include <sycl/stream.hpp>

using namespace sycl;

int main() {
  {
    queue Queue;

    // CHECK: <--- urMemRelease
    Queue.submit([&](handler &CGH) {
      stream Out(1024, 80, CGH);
      CGH.parallel_for<class test_cleanup1>(
          range<1>(2), [=](id<1> i) { Out << "Hello, World!" << endl; });
    });
    Queue.wait();
  }

  return 0;
}
