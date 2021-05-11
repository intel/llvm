// RUN: %clang_cc1 -disable-llvm-passes -triple spir64-unknown-unknown-sycldevice \
// RUN:   -fsycl-is-device -S -emit-llvm %s -o - | \
// RUN:   FileCheck %s

// The test checks that:
// 1. !sycl_explicit_simd metadata is generated for functions
// 2. !intel_reqd_sub_group_size !1 is added to explicit SIMD
//    kernel
// 3. Proper module !spirv.Source metadata is generated

template <typename name, typename Func>
void kernel(const Func &f) __attribute__((sycl_kernel)) {
  f();
}

void bar() {
  kernel<class MyKernel>([=]() __attribute__((sycl_explicit_simd)){});
  // CHECK: define {{.*}}spir_kernel void @_ZTSZ3barvE8MyKernel() {{.*}} !sycl_explicit_simd ![[EMPTY:[0-9]+]] !intel_reqd_sub_group_size ![[REQD_SIZE:[0-9]+]]

  kernel<class MyEsimdKernel>([=]() [[intel::sycl_explicit_simd]]{});
  // CHECK: define {{.*}}spir_kernel void @_ZTSZ3barvE13MyEsimdKernel() {{.*}} !sycl_explicit_simd ![[EMPTY:[0-9]+]] !intel_reqd_sub_group_size ![[REQD_SIZE]]
}

// CHECK: !spirv.Source = !{[[LANG:![0-9]+]]}
// CHECK: [[LANG]] = !{i32 0, i32 {{[0-9]+}}}
// CHECK: ![[EMPTY]] = !{}
// CHECK: ![[REQD_SIZE]] = !{i32 1}
