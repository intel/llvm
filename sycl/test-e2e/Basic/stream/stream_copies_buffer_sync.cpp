// RUN: %clangxx -fsycl -fsycl-targets=%sycl_triple %s -o %t.out
// RUN: %CPU_RUN_PLACEHOLDER %t.out %CPU_CHECK_PLACEHOLDER
// RUN: %GPU_RUN_ON_LINUX_PLACEHOLDER %t.out %GPU_CHECK_ON_LINUX_PLACEHOLDER
// RUN: %ACC_RUN_PLACEHOLDER %t.out %ACC_CHECK_PLACEHOLDER

#include <sycl/sycl.hpp>

using namespace sycl;

void dev_func(stream out) { out << "dev_func print\n"; }

int main() {
  {
    queue Queue;

    Queue.submit([&](handler &cgh) {
      stream stream1(1024, 160, cgh);
      cgh.parallel_for<class test_dev_func_stream>(
          nd_range<1>(range<1>(1), range<1>(1)), [=](sycl::nd_item<1> it) {
            stream1 << "stream1 print 1\n";
            dev_func(stream1);
            sycl::stream stream2 = stream1;
            stream1 << "stream1 print 2" << sycl::endl;
            stream2 << "stream2 print 1\n";
            stream1 << "stream1 print 3\n";
            stream2 << sycl::flush;
          });
    });
    Queue.wait();
    // CHECK: stream1 print 1
    // CHECK-NEXT: dev_func print
    // CHECK-NEXT: stream1 print 2
    // CHECK-NEXT: stream2 print 1
    // CHECK-NEXT: stream1 print 3
  }

  return 0;
}
