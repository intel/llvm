// RUN: %clang_cc1 -triple spir-unknown-unknown -O1 -cl-std=CL2.0 -fdeclare-opencl-builtins -finclude-default-header -emit-llvm-bc %s -o %t.bc
// RUN: llvm-spirv --spirv-ext=+SPV_EXT_image_raw10_raw12 %t.bc -o %t.spv
// RUN: llvm-spirv --spirv-ext=+SPV_EXT_image_raw10_raw12 %t.spv -to-text -o %t.spt
// RUN: FileCheck < %t.spt %s --check-prefix=CHECK-SPIRV

// RUN: llvm-spirv -r %t.spv -o %t.rev.bc
// RUN: llvm-dis < %t.rev.bc | FileCheck %s --check-prefixes=CHECK-COMMON,CHECK-LLVM
// RUN: llvm-spirv -r %t.spv --spirv-target-env=SPV-IR -o %t.rev.bc
// RUN: llvm-dis < %t.rev.bc | FileCheck %s --check-prefixes=CHECK-COMMON,CHECK-SPV-IR

// RUN: not llvm-spirv --spirv-ext=-SPV_EXT_image_raw10_raw12 %t.bc -o %t.spv 2>&1 | FileCheck %s --check-prefix=CHECK-EXT-OFF
// CHECK-EXT-OFF: Feature requires the following SPIR-V extension
// CHECK-EXT-OFF-NEXT: SPV_EXT_image_raw10_raw12

// CHECK-SPIRV: Extension "SPV_EXT_image_raw10_raw12"

// CHECK-COMMON: test_raw1012
// CHECK-LLVM: _Z27get_image_channel_data_type14ocl_image2d_ro
// CHECK-SPV-IR: call spir_func i32 @_Z24__spirv_ImageQueryFormatPU3AS133__spirv_Image__void_1_0_0_0_0_0_0
// CHECK-COMMON: switch i32
// CHECK-COMMON: i32 4323,
// CHECK-COMMON: i32 4324,
// CHECK-COMMON: icmp eq i32 %{{.*}}, 4323
// CHECK-COMMON: icmp eq i32 %{{.*}}, 4324

kernel void test_raw1012(global int *dst, read_only image2d_t img) {
  switch (get_image_channel_data_type(img)) {
  case CLK_SNORM_INT8:
    *dst = 8;
    break;
  case CLK_UNSIGNED_INT_RAW10_EXT:
    *dst = 10;
    break;
  case CLK_UNSIGNED_INT_RAW12_EXT:
    *dst = 12;
    break;
  }

  if (get_image_channel_data_type(img) == CLK_UNSIGNED_INT_RAW10_EXT)
    *dst = 1010;
  else if (CLK_UNSIGNED_INT_RAW12_EXT == get_image_channel_data_type(img))
    *dst = 1212;
}
