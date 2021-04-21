// RUN: %clang_cc1 -disable-llvm-passes -triple spir64-unknown-unknown-sycldevice -fsycl-is-device -S -emit-llvm %s -o - | FileCheck %s --check-prefixes CHECK,CHECK-ESIMD

// This test checks that attribute !intel_reqd_sub_group_size !1
// is added for kernels with !sycl_explicit_simd

void shared_func() { }

__attribute__((sycl_device)) __attribute__((sycl_explicit_simd)) void esimd_func() { shared_func(); }

// CHECK-ESIMD-DAG: define {{.*}}spir_kernel void @{{.*}}kernel_cm() #{{[0-9]+}} !sycl_explicit_simd !{{[0-9]+}} !intel_reqd_sub_group_size ![[SGSIZE1:[0-9]+]] {{.*}}{
// CHECK-ESIMD-DAG: define {{.*}}spir_func void @{{.*}}esimd_funcv() #{{[0-9]+}} !sycl_explicit_simd !{{[0-9]+}} !intel_reqd_sub_group_size ![[SGSIZE1]] {
// CHECK-ESIMD-DAG: define {{.*}}spir_func void @{{.*}}shared_funcv() #{{[0-9]+}} {
// CHECK-ESIMD-DAG: define linkonce_odr spir_func void @_ZN12ESIMDFunctorclEv({{.*}}) #{{[0-9]+}} {{.*}} !sycl_explicit_simd !{{[0-9]+}} {

class ESIMDFunctor {
public:
  void operator()() __attribute__((sycl_explicit_simd)) {
    esimd_func();
  }
};

template <typename name, typename Func>
__attribute__((sycl_kernel)) void kernel(Func kernelFunc) {
  kernelFunc();
}

void bar() {
  ESIMDFunctor cmf;
  kernel<class kernel_cm>(cmf);
}

// CHECK: !spirv.Source = !{[[LANG:![0-9]+]]}
// CHECK: [[LANG]] = !{i32 0, i32 {{[0-9]+}}}
// CHECK-ESIMD: ![[SGSIZE1]] = !{i32 1}
