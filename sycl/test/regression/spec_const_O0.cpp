// RUN: %clangxx -O0 -fsycl -fsycl-device-only -Xclang -emit-llvm %s -o - | FileCheck %s
// This test checks that even under -O0 the entire call chain starting from
// kh.get_specialization_constant and ending with
// __sycl_getScalar2020SpecConstantValue gets inlined.
#include <sycl/sycl.hpp>

using namespace sycl;

constexpr specialization_id<unsigned int> SPEC_CONST(1024);

SYCL_EXTERNAL unsigned int foo(kernel_handler &kh) {
  return kh.get_specialization_constant<SPEC_CONST>();
}

// CHECK-LABEL: define dso_local spir_func noundef i32 @_Z3foo{{.*}} {
// CHECK-NOT: {{.*}}spir_func
// CHECK: %{{.*}} = {{.*}}call spir_func {{.*}}i32 @_Z37__sycl_getScalar2020SpecConstantValue
// CHECK-NOT: {{.*}}spir_func
// CHECK-LABEL: }
