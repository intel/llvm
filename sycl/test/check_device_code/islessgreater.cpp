// RUN: %clangxx -I %sycl_include -S -emit-llvm -fsycl-device-only %s -o - -Xclang -disable-llvm-passes | FileCheck %s

#include <CL/sycl.hpp>

SYCL_EXTERNAL void test_islessgreater(float a, float b) {
  sycl::islessgreater(a, b);
}
// CHECK-NOT: __spirv_LessOrGreater
// CHECK: {{.*}} = call spir_func noundef zeroext i1 @_Z20__spirv_FOrdNotEqualff(float {{.*}}, float {{.*}})
