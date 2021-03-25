// TODO: previously code generation and ESIMD lowering was
// a part of the same %clang_cc1 invocation, but now it is
// separate. So, we can split this test into 2, where one
// will be testing code generation and the second ESIMD lowering.
//
// RUN: %clang_cc1 -disable-llvm-passes -triple spir64-unknown-unknown-sycldevice \
// RUN:   -fsycl-is-device -fsycl-explicit-simd -emit-llvm %s -o %t
// RUN: sycl-post-link -split-esimd -lower-esimd -O0 -S %t -o %t.table
// RUN: FileCheck %s -input-file=%t_esimd_0.ll

// This test checks that FE allows globals with register_num attribute in ESIMD mode.

__attribute__((opencl_private)) __attribute__((sycl_explicit_simd)) __attribute__((register_num(17))) int vc;
// CHECK: @vc = {{.+}} i32 0, align 4 #0

template <typename name, typename Func>
__attribute__((sycl_kernel)) void kernel(Func kernelFunc) {
  kernelFunc();
}

void init_vc(int x) {
  kernel<class kernel_esimd>([=]() __attribute__((sycl_explicit_simd)) {
    vc = x;
    // CHECK: store i32 %{{[0-9a-zA-Z_]+}}, i32* @vc
  });
}
// CHECK: attributes #0 = {{.*"VCByteOffset"="17".*"VCVolatile"}}
