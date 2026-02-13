// RUN: not %clangxx -fsycl -fsycl-device-only -Xclang -fsycl-allow-func-ptr -S %s -o /dev/null 2>&1 -DRET_REF | FileCheck -check-prefix CHECK-RET %s
// RUN: not %clangxx -fsycl -fsycl-device-only -Xclang -fsycl-allow-func-ptr -S %s -o /dev/null 2>&1 | FileCheck -check-prefix CHECK-ARG %s
#include <sycl/ext/oneapi/experimental/invoke_simd.hpp>
#include <sycl/sycl.hpp>

using namespace sycl::ext::oneapi::experimental;
using namespace sycl;
namespace esimd = sycl::ext::intel::esimd;
#ifdef RET_REF
const int a = 0;
[[intel::device_indirectly_callable]] const int &callee() { return a; }
#else
[[intel::device_indirectly_callable]] void callee(double, float, short &) {}
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
#ifdef RET_REF
      invoke_simd(ndi.get_sub_group(), callee);
#else
       short s;
       invoke_simd(ndi.get_sub_group(), callee, 0, 0, s);
#endif
    });
  });
}

int main() {
  foo();
  // CHECK-ARG: {{.*}}error:{{.*}}static assertion failed due to requirement '!callable_has_ref_arg': invoke_simd does not support callables with reference arguments
  // CHECK-RET: {{.*}}error:{{.*}}static assertion failed due to requirement '!callable_has_ref_ret': invoke_simd does not support callables returning references
}
