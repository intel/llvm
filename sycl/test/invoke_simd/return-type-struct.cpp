// RUN: not %clangxx -fsycl -fsycl-device-only -Xclang -fsycl-allow-func-ptr -S %s -o /dev/null 2>&1 | FileCheck %s
// RUN: %clangxx -fsycl -fsycl-device-only -Xclang -fsycl-allow-func-ptr -S %s -o /dev/null -DUNIFORM
#include <sycl/ext/oneapi/experimental/invoke_simd.hpp>
#include <sycl/sycl.hpp>

struct Foo {};
using namespace sycl::ext::oneapi::experimental;
using namespace sycl;
namespace esimd = sycl::ext::intel::esimd;

#ifdef UNIFORM
[[intel::device_indirectly_callable]] uniform<Foo> callee(simd<int, 8>) {
  return uniform(Foo{});
}
#else
[[intel::device_indirectly_callable]] Foo callee(simd<int, 8>) { return Foo{}; }
#endif
void foo() {
  constexpr unsigned Size = 1024;
  constexpr unsigned GroupSize = 64;
  sycl::range<1> GlobalRange{Size};
  sycl::range<1> LocalRange{GroupSize};
  sycl::nd_range<1> Range(GlobalRange, LocalRange);
  queue q;
  auto e = q.submit([&](handler &cgh) {
    cgh.parallel_for(Range, [=](nd_item<1> ndi) {
      auto x = invoke_simd(ndi.get_sub_group(), callee, 0);
    });
  });
}

int main() {
  foo();
  // CHECK: {{.*}}error:{{.*}}static assertion failed due to requirement '!callable_has_non_uniform_struct_ret': invoke_simd does not support callables returning non-uniform structures{{.*}}
}
