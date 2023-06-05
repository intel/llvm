// RUN: %{build} -O0 -o %t.O0.out
// RUN: %{run} %t.O0.out
// RUN: %{build} -O1 -o %t.O1.out
// RUN: %{run} %t.O1.out
// RUN: %{build} -O2 -o %t.O2.out
// RUN: %{run} %t.O2.out
// RUN: %{build} -O3 -o %t.O3.out
// RUN: %{run} %t.O3.out
// RUN: %{build} -Ofast -o %t.Ofast.out
// RUN: %{run} %t.Ofast.out
// REQUIRES: linux
// UNSUPPORTED: hip || cuda

#include <sycl/sycl.hpp>
using namespace sycl;
static constexpr unsigned N = 16;
static constexpr int scalar = 8;

template <typename T>
void stream_triad(std::array<T, N> &A, std::array<T, N> &B, std::array<T, N> &C) {
  auto q = queue{};
  device d = q.get_device();
  std::cout << "Using " << d.get_info<info::device::name>() << "\n";
  auto range = sycl::range<1>{N};

  {
    auto bufA = buffer<T, 1>{A.data(), range};
    auto bufB = buffer<T, 1>{B.data(), range};
    auto bufC = buffer<T, 1>{C.data(), range};
    q.submit([&](handler &cgh) {
      auto A = bufA.template get_access<access::mode::write>(cgh);
      auto B = bufB.template get_access<access::mode::read>(cgh);
      auto C = bufC.template get_access<access::mode::read>(cgh);
      cgh.parallel_for<class kernel_stream_triad>(range, [=](sycl::id<1> id) {
        A[id] = B[id] + C[id] * scalar;
      });
    });
  }
}

int main() {
  std::array<int, N> A{0};
  std::array<int, N> B{0};
  std::array<int, N> C{0};
  for (unsigned i = 0; i < N; ++i) {
    B[i] = C[i] = i;
  }
  stream_triad(A, B, C);
  for (unsigned i = 0; i < N; ++i) {
    assert(A[i] == (i + i * scalar));
  }
  std::cout << "Test passed" << std::endl;
}
