// RUN: %clangxx -fsycl %s -o %t.out
// RUN: env SYCL_DEVICE_TYPE=HOST %t.out
// RUN: %CPU_RUN_PLACEHOLDER %t.out
// RUN: %GPU_RUN_PLACEHOLDER %t.out
// RUN: %ACC_RUN_PLACEHOLDER %t.out

#include <CL/sycl.hpp>

namespace sycl = cl::sycl;

int main() {
  sycl::queue().submit([&](sycl::handler &h) {
    h.single_task<class task>([=]() {
      sycl::vec<float, 1> A{1}, B{2}, C{3};
      sycl::vec<float, 1> res = sycl::mad(A, B, C);
      assert(res.x() - 5.0f < 1e-5);
    });
  });
}