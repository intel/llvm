// RUN: %clang_cc1 -disable-llvm-passes -triple spir64-unknown-unknown-sycldevice \
// RUN:   -fsycl-is-device -emit-llvm %s -o - | FileCheck %s

// This test checks that FE generates appropriate attributes for ESIMD private globals with register_num attribute.

__attribute__((opencl_private)) __attribute__((sycl_explicit_simd)) __attribute__((register_num(17))) int vc;
// CHECK: @vc = {{.+}} i32 0, align 4 #[[ATTR:[0-9]+]]

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
// CHECK: attributes #[[ATTR]] = { "genx_byte_offset"="17" "genx_volatile" }
