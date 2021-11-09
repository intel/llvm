// REQUIRES: cuda || hip_be
// RUN: %clangxx -fsycl -fsycl-targets=%sycl_triple,spir64 %s -o -
// RUN: %clangxx -fsycl -fsycl-targets=spir64,%sycl_triple %s -o -
//
// RUN: %clangxx -fsycl -fsycl-device-code-split=per_kernel -fsycl-targets=%sycl_triple,spir64 %s -o -
// RUN: %clangxx -fsycl -fsycl-device-code-split=per_kernel -fsycl-targets=spir64,%sycl_triple %s -o -
//
// Test checks that compiling for multiple devices works regardless of target
// order.

#include <sycl/sycl.hpp>

using namespace cl::sycl;

int main() {
  sycl::queue q;

  float A_Data[5] = {1.1};
  float B_Data[5] = {0};
  int C_Data[10] = {0};

  {
    buffer<float, 1> A_buff(A_Data, range<1>(5));
    buffer<float, 1> B_buff(B_Data, range<1>(5));
    q.submit([&](handler &cgh) {
       auto A_acc = A_buff.get_access<access::mode::read>(cgh);
       auto B_acc = B_buff.get_access<access::mode::write>(cgh);
       cgh.parallel_for(range<1>{5},
                        [=](id<1> index) { B_acc[index] = A_acc[index]; });
     }).wait();
  }

  {
    buffer<int, 1> C_buff(C_Data, range<1>(10));
    q.submit([&](handler &cgh) {
       auto C_acc = C_buff.get_access<access::mode::write>(cgh);
       cgh.parallel_for(range<1>{10}, [=](id<1> index) { C_acc[index] = 15; });
     }).wait();
  }
}
