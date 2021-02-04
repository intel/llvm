// REQUIRES: gpu,level_zero
// RUN: %clangxx -fsycl -fsycl-targets=%sycl_triple %s -o %t.out
// RUN: env ZE_DEBUG=1 %GPU_RUN_PLACEHOLDER %t.out 2> %t1.out; cat %t1.out %GPU_CHECK_PLACEHOLDER

#include <CL/sycl.hpp>

using namespace cl::sycl;

int main() {
  constexpr int Size = 100;
  {
    queue Queue;
    buffer<::cl_int, 1> Buffer(Size);

    Queue.submit([&](handler &cgh) {
      accessor Accessor{Buffer, cgh, read_write};
      cgh.parallel_for<class CreateBuffer>(range<1>(Size), [=](id<1> ID) {});
    });
    Queue.wait();
  }

  return 0;
}

// CHECK: Buffer Create: {{Integrated|Discrete}} GPU will use [[API:zeMemAllocHost|zeMemAllocDevice]]
// CHECK-NEXT: ZE ---> [[API]](
