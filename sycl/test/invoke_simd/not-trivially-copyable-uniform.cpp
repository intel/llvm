// RUN: not %clangxx -fsycl -fsycl-device-only -Xclang -fsycl-allow-func-ptr -S %s -o /dev/null 2>&1 | FileCheck -check-prefix CHECK-ARG %s
// RUN: not %clangxx -fsycl -fsycl-device-only -Xclang -fsycl-allow-func-ptr -DRET -S %s -o /dev/null 2>&1 | FileCheck -check-prefix CHECK-RET %s

#include <sycl/ext/oneapi/experimental/invoke_simd.hpp>
#include <sycl/sycl.hpp>

using namespace sycl::ext::oneapi::experimental;
using namespace sycl;
namespace esimd = sycl::ext::intel::esimd;
struct B {
  virtual ~B() {}
};
struct D : public B {
  ~D() override {}
};

#ifdef RET
[[intel::device_indirectly_callable]] uniform<D> callee() {}
#else
[[intel::device_indirectly_callable]] void callee(D d) {}
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
#ifdef RET
      invoke_simd(ndi.get_sub_group(), callee);
#else
      D d;
      invoke_simd(ndi.get_sub_group(), callee, uniform{d});
#endif
    });
  });
}

int main() {
  foo();
  // CHECK-ARG: {{.*}}error:{{.*}}static assertion failed due to requirement '!has_non_trivially_copyable_uniform_arg': Uniform arguments must be trivially copyable
  // CHECK-RET: {{.*}}error:{{.*}}static assertion failed due to requirement '!callable_has_uniform_non_trivially_copyable_ret': invoke_simd does not support callables returning uniforms that are not trivially copyable
}
