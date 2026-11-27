// REQUIRES: native_cpu_ock
// RUN: %clangxx -fsycl -fsycl-targets=native_cpu -Xclang -fsycl-allow-func-ptr -O2 -mllvm -sycl-native-dump-device-ir %s -o %t.out | FileCheck %s

// Check that vecz handles an indirect call whose callee varies between
// work-items: each lane must call its own function pointer with its own
// argument, rather than one callee being used for the whole vector.

#include <sycl/sycl.hpp>

using fn_t = int (*)(int);
SYCL_EXTERNAL int add1(int x) { return x + 1; }
SYCL_EXTERNAL int mul2(int x) { return x * 2; }

int main() {
  sycl::queue q;
  sycl::buffer<int, 1> sel(sycl::range<1>(16));
  sycl::buffer<int, 1> out(sycl::range<1>(16));
  q.submit([&](sycl::handler &h) {
    sycl::accessor s(sel, h, sycl::read_only);
    sycl::accessor o(out, h, sycl::write_only);
    h.parallel_for<class IndirectCall>(sycl::range<1>(16), [=](sycl::id<1> i) {
      fn_t f = s[i] ? add1 : mul2;
      o[i] = f(i[0]);
    });
  });
}

// CHECK-LABEL: define void @_ZTSZZ4mainENKUlRN4sycl3_V17handlerEE_clES2_E12IndirectCall.NativeCPUKernel(
// CHECK: [[FNS:%.*]] = select <8 x i1> %{{.*}}, <8 x ptr> {{.*}}@_Z4mul2i{{.*}}, <8 x ptr> {{.*}}@_Z4add1i
// CHECK-DAG: [[F0:%.*]] = extractelement <8 x ptr> [[FNS]], i64 0
// CHECK-DAG: [[F7:%.*]] = extractelement <8 x ptr> [[FNS]], i64 7
// CHECK-DAG: [[A0:%.*]] = extractelement <8 x i32> [[ARGS:%.*]], i64 0
// CHECK-DAG: [[A7:%.*]] = extractelement <8 x i32> [[ARGS]], i64 7
// CHECK: call noundef i32 [[F0]](i32 noundef [[A0]])
// CHECK: call noundef i32 [[F7]](i32 noundef [[A7]])
