// RUN: %clang_cc1 -triple spir-unknown-unknown -O1 -cl-std=CL2.0 -emit-llvm-bc %s -o %t.bc
// RUN: llvm-spirv --spirv-ext=+SPV_EXT_image_raw10_raw12 %t.bc -o %t.spv
// RUN: llvm-spirv --spirv-ext=+SPV_EXT_image_raw10_raw12 %t.spv -to-text -o %t.spt
// RUN: FileCheck < %t.spt %s --check-prefix=CHECK-SPIRV

// CHECK-SPIRV-NOT: Extension "SPV_EXT_image_raw10_raw12"

// Test that use of constant values equal to the extension's constants do not enable the extension.

kernel void test_raw1012(global int *dst, int value) {
  switch (value) {
  case 0x10E3: // same value as CLK_UNSIGNED_INT_RAW10_EXT
    *dst = 10;
    break;
  case 0x10E4: // same value as CLK_UNSIGNED_INT_RAW12_EXT
    *dst = 12;
    break;
  }

  if (value==0x10E3 || value==0x10E4) {
    *dst = 1012;
  }
}
