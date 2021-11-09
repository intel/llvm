// RUN: %clang_cc1 -triple x86_64-windows-msvc -fdeclare-spirv-builtins -fsyntax-only -emit-llvm %s -o - | FileCheck %s

float acos(float val) {
  // CHECK: @"?acos@@YAMM@Z"
  // CHECK: call float @"?__spirv_ocl_acos@@YAMM@Z"
  return __spirv_ocl_acos(val);
}

// CHECK: declare dso_local float @"?__spirv_ocl_acos@@YAMM@Z"(float)

double acos(double val) {
  // CHECK: @"?acos@@YANN@Z"
  // CHECK: call double @"?__spirv_ocl_acos@@YANN@Z"
  return __spirv_ocl_acos(val);
}

// CHECK: declare dso_local double @"?__spirv_ocl_acos@@YANN@Z"(double)
