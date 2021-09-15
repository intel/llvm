// RUN: %clangxx -I %sycl_include -S -emit-llvm -fsycl-device-only %s -o - -Xclang -disable-llvm-passes | FileCheck %s

#include <CL/sycl.hpp>

using namespace sycl;

SYCL_EXTERNAL void test_group_mask(sub_group g) {
  ext::oneapi::group_ballot(g, true);
}
// CHECK: %{{.*}} =  call spir_func <4 x i32> @_Z[[#]]__spirv_GroupNonUniformBallotjb(i32 {{.*}}, i1{{.*}})
