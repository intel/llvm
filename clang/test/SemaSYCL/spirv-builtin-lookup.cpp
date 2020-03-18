// RUN: %clang_cc1 -fdeclare-spirv-builtins -fsyntax-only -verify -std=c++11 %s
// expected-no-diagnostics

// Verify that __spirv_ocl_acos is recognized as a builtin

float acos(float val) {
  return __spirv_ocl_acos(val);
}

double acos(double val) {
  return __spirv_ocl_acos(val);
}
