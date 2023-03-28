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
static constexpr unsigned N = 16;

template <typename T>
void stream_copy(std::array<T, N> &A, std::array<T, N> &B) {
  auto q = queue{};
  device d = q.get_device();
  std::cout << "Using " << d.get_info<info::device::name>() << "\n";
  auto range = sycl::range<1>{N};

  {
    auto bufA = buffer<T, 1>{A.data(), range};
    auto bufB = buffer<T, 1>{B.data(), range};
    q.submit([&](handler &cgh) {
      auto A = bufA.template get_access<access::mode::write>(cgh);
      auto B = bufB.template get_access<access::mode::read>(cgh);
      cgh.parallel_for<class kernel_stream_copy>(range, [=](sycl::id<1> id) {
        A[id] = B[id];
      });
    });
  }
}

int main() {
  std::array<int, N> A{0};
  std::array<int, N> B{0};
  for (unsigned i = 0; i < N; ++i) {
    B[i] = i;
  }
  stream_copy(A, B);
  for (unsigned i = 0; i < N; ++i) {
    assert(A[i] == i);
  }
  std::cout << "Test passed" << std::endl;
}
