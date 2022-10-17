// RUN: %clang_cc1 -triple x86_64-windows-msvc -fdeclare-spirv-builtins -fsyntax-only -emit-llvm %s -o - | FileCheck %s

float acos(float val) {
  // CHECK: @"?acos@@YAMM@Z"
  // CHECK: call noundef float @"?__spirv_ocl_acos@@YAMM@Z"
  return __spirv_ocl_acos(val);
}

// CHECK: declare dso_local noundef float @"?__spirv_ocl_acos@@YAMM@Z"(float noundef)

double acos(double val) {
  // CHECK: @"?acos@@YANN@Z"
  // CHECK: call noundef double @"?__spirv_ocl_acos@@YANN@Z"
  return __spirv_ocl_acos(val);
}

// CHECK: declare dso_local noundef double @"?__spirv_ocl_acos@@YANN@Z"(double noundef)
