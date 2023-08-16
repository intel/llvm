// RUN: not %clangxx -fsycl -fsycl-device-only -S %s -o /dev/null 2>&1 | FileCheck %s
#include <sycl/ext/oneapi/experimental/invoke_simd.hpp>
#include <sycl/sycl.hpp>

using namespace sycl::ext::oneapi::experimental;
using namespace sycl;
namespace esimd = sycl::ext::intel::esimd;

[[intel::device_indirectly_callable]] void
callee(simd<int, 8>, std::tuple<simd<int, 4>, simd<int, 4>>) {
  return std::make_tuple(simd<int, 4>(), simd<int, 4>());
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
      std::tuple<simd<int, 4>, simd<int, 4>> arg;
      invoke_simd(ndi.get_sub_group(), callee, 0, arg);
    });
  });
}

int main() {
  foo();
  // CHECK: {{.*}}error:{{.*}}static assertion failed due to requirement '!callable_has_struct_arg': invoke_simd does not support callables with structure arguments{{.*}}
}
