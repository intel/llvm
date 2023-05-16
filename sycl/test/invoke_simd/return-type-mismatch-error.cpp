// RUN: not %clangxx -fsycl -fsycl-device-only -S %s -o /dev/null 2>&1 | FileCheck %s
#include <sycl/ext/oneapi/experimental/invoke_simd.hpp>
#include <sycl/sycl.hpp>

using namespace sycl::ext::oneapi::experimental;
using namespace sycl;
namespace esimd = sycl::ext::intel::esimd;

[[intel::device_indirectly_callable]] simd<int, 4> callee(simd<int, 8>) {
  return simd<int, 4>();
}

void foo() {
  constexpr unsigned Size = 1024;
  constexpr unsigned GroupSize = 64;
  sycl::range<1> GlobalRange{Size};
  sycl::range<1> LocalRange{GroupSize};
  sycl::nd_range<1> Range(GlobalRange, LocalRange);
  queue q;
  auto e = q.submit([&](handler &cgh) {
    cgh.parallel_for(Range, [=](nd_item<1> ndi) {
      invoke_simd(ndi.get_sub_group(), callee, 0);
    });
  });
}

int main() {
  foo();
  // CHECK: {{.*}}error:{{.*}}static assertion failed due to requirement 'RetVecLength == 8': invoke_simd callee return type vector length must match kernel subgroup size{{.*}}
}
