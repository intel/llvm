// RUN: %clang_cc1 -fdeclare-spirv-builtins -fsyntax-only -verify %s
// expected-no-diagnostics

// Verify that __spirv_ocl_acos is recognized as a builtin

float acos(float val) {
  return __spirv_ocl_acos(val);
}

double acos(double val) {
  return __spirv_ocl_acos(val);
}

typedef int int4 __attribute__((ext_vector_type(4)));
typedef float float4 __attribute__((ext_vector_type(4)));

int4 ilogb() {
  float4 f4 = {0.f, 0.f, 0.f, 0.f};
  int4 i4 = __spirv_ocl_ilogb(f4);
  return i4;
}

double sincos(double val, double *res) {
  return __spirv_ocl_sincos(val, res);
}

double dot(float4 v1, float4 v2) {
  return __spirv_Dot(v1, v2);
}
