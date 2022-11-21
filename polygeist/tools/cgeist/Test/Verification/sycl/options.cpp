// COM: Ensure warnings are suppressed (-w)
// RUN: clang++ -w -fsycl -fsycl-device-only -fsycl-targets=spir64-unknown-unknown-syclmlir %s -S -emit-llvm -o - 2>&1 | FileCheck %s --implicit-check-not="{{warning|Warning}}:"

#include <sycl/sycl.hpp>

using namespace sycl;
static constexpr unsigned N = 16;

// CHECK-LABEL: define weak_odr spir_kernel void @_ZTSZZ12parallel_forRSt5arrayIiLm16EEENKUlRN4sycl3_V17handlerEE_clES5_EUlNS3_2idILi1EEEE

void parallel_for(std::array<int, N> &A) {
  auto q = queue{};
  device d = q.get_device();
  auto range = sycl::range<1>{N};
  {
    auto bufA = buffer<int, 1>{A.data(), range};
    q.submit([&](handler &cgh) {
      accessor A(bufA, cgh);
      cgh.parallel_for(range, [=](id<1> id) { A[id] = id; });
    });
  }
}

int main() {
  std::array<int, N> A{0};
  parallel_for(A);
  return A[0];
}

