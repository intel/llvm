// RUN: %clangxx -fsycl -fsycl-targets=%sycl_triple -O0 -w %s -o %t.O0.out
// RUN: %CPU_RUN_PLACEHOLDER %t.O0.out
// RUN: %GPU_RUN_PLACEHOLDER %t.O0.out
// RUN: %ACC_RUN_PLACEHOLDER %t.O0.out
// RUN: %clangxx -fsycl -fsycl-targets=%sycl_triple -O1 -w %s -o %t.O1.out
// RUN: %CPU_RUN_PLACEHOLDER %t.O1.out
// RUN: %GPU_RUN_PLACEHOLDER %t.O1.out
// RUN: %ACC_RUN_PLACEHOLDER %t.O1.out
// RUN: %clangxx -fsycl -fsycl-targets=%sycl_triple -O2 -w %s -o %t.O2.out
// RUN: %CPU_RUN_PLACEHOLDER %t.O2.out
// RUN: %GPU_RUN_PLACEHOLDER %t.O2.out
// RUN: %ACC_RUN_PLACEHOLDER %t.O2.out
// RUN: %clangxx -fsycl -fsycl-targets=%sycl_triple -O3 -w %s -o %t.O3.out
// RUN: %CPU_RUN_PLACEHOLDER %t.O3.out
// RUN: %GPU_RUN_PLACEHOLDER %t.O3.out
// RUN: %ACC_RUN_PLACEHOLDER %t.O3.out
// RUN: %clangxx -fsycl -fsycl-targets=%sycl_triple -Ofast -w %s -o %t.Ofast.out
// RUN: %CPU_RUN_PLACEHOLDER %t.Ofast.out
// RUN: %GPU_RUN_PLACEHOLDER %t.Ofast.out
// RUN: %ACC_RUN_PLACEHOLDER %t.Ofast.out
// REQUIRES: linux
// UNSUPPORTED: hip || cuda

#include <sycl/sycl.hpp>
using namespace sycl;

void single_task(std::array<int, 1> &A) {
  auto q = queue{};
  device d = q.get_device();
  std::cout << "Using " << d.get_info<info::device::name>() << "\n";

  {
    auto buf = buffer<int, 1>{A.data(), 1};
    q.submit([&](handler &cgh) {
      auto A = buf.get_access<access::mode::write>(cgh);
      cgh.single_task<class kernel_single_task>([=]() {
        A[0] = 1;
      });
    });
  }
}

int main() {
  std::array<int, 1> A = {0};
  single_task(A);
  assert(A[0] == 1);
  std::cout << "Test passed" << std::endl;
}
