// RUN: not %clangxx -fsycl -fsycl-device-only -Xclang -fsycl-allow-func-ptr -S %s -o /dev/null 2>&1 | FileCheck %s
// RUN: %clangxx -fsycl -fsycl-device-only -Xclang -fsycl-allow-func-ptr -S %s -o /dev/null -DUNIFORM

#include <sycl/ext/oneapi/experimental/invoke_simd.hpp>
#include <sycl/sycl.hpp>

using namespace sycl::ext::oneapi::experimental;
using namespace sycl;
namespace esimd = sycl::ext::intel::esimd;
struct Foo {};
[[intel::device_indirectly_callable]] void callee(simd<int, 8>, Foo) {}

void foo() {
  constexpr unsigned Size = 1024;
  constexpr unsigned GroupSize = 64;
  sycl::range<1> GlobalRange{Size};
  sycl::range<1> LocalRange{GroupSize};
  sycl::nd_range<1> Range(GlobalRange, LocalRange);
  queue q;
  auto e = q.submit([&](handler &cgh) {
    cgh.parallel_for(Range, [=](nd_item<1> ndi) {
      Foo arg;
#ifdef UNIFORM
      invoke_simd(ndi.get_sub_group(), callee, 0, uniform{arg});
#else
      invoke_simd(ndi.get_sub_group(), callee, 0, arg);
#endif
    });
  });
}

int main() {
  foo();
  // CHECK: {{.*}}error:{{.*}}static assertion failed due to requirement '!has_non_uniform_struct_arg': Structure arguments must be uniform{{.*}}
}
