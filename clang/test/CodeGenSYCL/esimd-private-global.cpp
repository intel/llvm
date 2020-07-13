// RUN: %clang_cc1 -disable-llvm-passes -triple spir64-unknown-unknown-sycldevice \
// RUN:   -fsycl -fsycl-is-device -fsycl-explicit-simd -emit-llvm %s -o - | \
// RUN:   FileCheck %s

// This test checks that FE allows globals with register_num attribute in ESIMD mode.

__attribute__((opencl_private)) __attribute__((register_num(17))) int vc;
// CHECK: @vc = {{.+}} i32 0, align 4 #0

SYCL_EXTERNAL void init_vc(int x) {
  vc = x;
  // CHECK: store i32 %x, i32* @vc
}
// CHECK: attributes #0 = { "genx_byte_offset"="17" "genx_volatile" }
