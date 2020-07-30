// REQUIRES: nvptx-registered-target
// RUN: %clang_cc1 -triple nvptx-nvidia-cuda -target-cpu sm_20 -S -o - %s | FileCheck %s -check-prefix=NOF16
// RUN: %clang_cc1 -triple nvptx-nvidia-cuda -target-cpu sm_60 -S -o - %s | FileCheck %s

// CHECK: .target sm_60
// NOF16: .target sm_20

void f() {
  _Float16 x, y, z;
  // CHECK: add.rn.f16
  // NOF16: add.rn.f32
  z = x + y;
  // CHECK: sub.rn.f16
  // NOF16: sub.rn.f32
  z = x - y;
  // CHECK: mul.rn.f16
  // NOF16: mul.rn.f32
  z = x * y;
  // CHECK: div.rn.f32
  // NOF16: div.rn.f32
  z = x / y;
}
