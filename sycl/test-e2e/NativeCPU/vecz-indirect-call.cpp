// REQUIRES: native_cpu
// RUN: %clangxx -fsycl -fsycl-targets=native_cpu -Xclang -fsycl-allow-func-ptr -O2 -mllvm -sycl-native-dump-device-ir %s | FileCheck %s

// Check that vecz handles indirect calls without crashing.

#include <sycl/sycl.hpp>

using fn_t = int (*)(int);
SYCL_EXTERNAL int add1(int x) { return x + 1; }
SYCL_EXTERNAL int mul2(int x) { return x * 2; }

int main() {
  sycl::queue q;
  const int wanted[16] = {
    0, 1, 0, 0, 
    1, 1, 1, 0, 
    0, 0, 1, 1, 
    1, 1, 0, 0
  };
  sycl::buffer<int, 1> sel(sycl::range<1>(16));
  {

    sycl::host_accessor s(sel, sycl::write_only);
    for (int i = 0; i < 16; ++i)
      s[i] = wanted[i];
  }
  sycl::buffer<int, 1> out(sycl::range<1>(16));
  q.submit([&](sycl::handler &h) {
    sycl::accessor s(sel, h, sycl::read_only);
    sycl::accessor o(out, h, sycl::write_only);
    h.parallel_for<class IndirectCall>(sycl::range<1>(16), [=](sycl::id<1> i) {
      fn_t f = s[i] ? add1 : mul2;
      o[i] = f(i[0]);
    });
  });
  sycl::host_accessor o(out, sycl::read_only);
  for (int i = 0; i < 16; ++i)
    assert(o[i] == (wanted[i] ? i + 1 : i * 2));
}

// CHECK-LABEL: define void @_ZTSZZ4mainENKUlRN4sycl3_V17handlerEE_clES2_E12IndirectCall.NativeCPUKernel(
// CHECK: [[ARG:%.*]] = extractelement <8 x i32>
// CHECK: call noundef i32 %{{.*}}(i32 noundef [[ARG]])