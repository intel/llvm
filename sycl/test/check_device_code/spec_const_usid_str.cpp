// RUN: %clangxx -fsycl -fsycl-device-only -fsycl-targets=spirv64 %s -S -emit-llvm -o - | FileCheck %s --check-prefixes=CHECK-SPIRV64
// RUN: %if hip %{ %clangxx -fsycl -fsycl-device-only -fsycl-targets=amdgcn-amd-amdhsa %s -S -emit-llvm -o - | FileCheck %s --check-prefixes=CHECK-HIP %}

// This test checks that __usid_str has constant address space.

// CHECK-SPIRV64:@__usid_str = private unnamed_addr addrspace(1) constant [38 x i8] c"uid
// CHECK-HIP: @__usid_str = private unnamed_addr addrspace(4) constant [38 x i8] c"uid

#include <sycl/sycl.hpp>

using namespace sycl;

constexpr specialization_id<unsigned int> SPEC_CONST(1024);

SYCL_EXTERNAL unsigned int foo(kernel_handler &kh) {
  return kh.get_specialization_constant<SPEC_CONST>();
}
