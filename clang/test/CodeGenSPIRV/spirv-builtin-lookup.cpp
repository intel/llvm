// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fdeclare-spirv-builtins -fsyntax-only -emit-llvm %s -o - | FileCheck %s

float acos(float val) {
  // CHECK: @_Z4acosf
  // CHECK: call noundef float @_Z16__spirv_ocl_acosf
  return __spirv_ocl_acos(val);
}

// CHECK: declare noundef float @_Z16__spirv_ocl_acosf(float noundef)

double acos(double val) {
  // CHECK: @_Z4acosd
  // CHECK: call noundef double @_Z16__spirv_ocl_acosd
  return __spirv_ocl_acos(val);
}

// CHECK: declare noundef double @_Z16__spirv_ocl_acosd(double noundef)
