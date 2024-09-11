// REQUIRES: nvptx-registered-target

// RUN: %clang -### -fsycl -fsycl-targets=nvptx64-nvidia-cuda %s 2>&1 | FileCheck %s

// Check that we correctly determine that the final link command links
// devicelibs together, as far as the driver is concerned. This results in the
// -only-needed flag.
//
// Note we check the names of the various device libraries because that's the
// logic the driver uses.

// CHECK: llvm-link

// CHECK: llvm-link
// CHECK-SAME: -only-needed
// CHECK-SAME: devicelib--cuda.bc
// CHECK-SAME: libspirv-nvptx64-nvidia-cuda.bc
// CHECK-SAME: libdevice

void func(){}
