// REQUIRES: level_zero || cuda

// RUN: %clangxx -fsycl -fsycl-targets=%sycl_triple %s -o %t2.out
// RUN: env SYCL_PI_TRACE=2 ZE_DEBUG=1 %GPU_RUN_PLACEHOLDER %t2.out 2>&1 %GPU_L0_CHECK_PLACEHOLDER
// RUN: %CPU_RUN_PLACEHOLDER %t2.out
// RUN: %GPU_RUN_PLACEHOLDER %t2.out
// RUN: %ACC_RUN_PLACEHOLDER %t2.out

#include <sycl/sycl.hpp>

#include <cassert>
#include <string>

using namespace sycl;

int main() {
  {
    int data1[10] = {-1, -1, -1, -1, -1, -1, -1, -1, -1, -1};
    {
      buffer<int, 1> a(data1, range<1>(10), {property::buffer::use_host_ptr()});
      buffer<int, 1> b(
          range<1>(10),
          {ext::oneapi::property::buffer::use_pinned_host_memory()});
      queue myQueue;
      myQueue.submit([&](handler &cgh) {
        auto A = a.get_access<access::mode::read_write>(cgh);
        auto B = b.get_access<access::mode::read_write>(cgh);
        cgh.parallel_for<class init_b>(range<1>{10}, [=](id<1> index) {
          B[index] = 0;
          A[index] = B[index] + 1;
        });
      });
    } // Data is copied back because there is a user side shared_ptr
    for (int i = 0; i < 10; i++)
      assert(data1[i] == 1);
  }
}

// CHECK:---> piMemBufferCreate
// CHECK:---> piMemBufferCreate
// CHECK-NEXT: {{.*}} : {{.*}}
// CHECK-NEXT: {{.*}} : {{.*}}
// CHECK-NEXT: {{.*}} : 17
