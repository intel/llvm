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
static constexpr unsigned N = 8;

void parallel_for(std::array<int, N> &A) {
  auto q = queue{};
  device d = q.get_device();
  std::cout << "Using " << d.get_info<info::device::name>() << "\n";
  auto range = sycl::range<1>{N};

  {
    auto buf = buffer<int, 1>{A.data(), range};
    q.submit([&](handler &cgh) {
      auto A = buf.get_access<access::mode::write>(cgh);
      cgh.parallel_for<class kernel_parallel_for>(range, [=](sycl::id<1> id) {
        A[id] = id;
      });
    });
  }
}

int main() {
  std::array<int, N> A{0};
  parallel_for(A);
  for (unsigned i = 0; i < N; ++i) {
    assert(A[i] == i);
  }
  std::cout << "Test passed" << std::endl;
}
