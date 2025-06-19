// REQUIRES: hip
// RUN: %clangxx -fsycl -fsycl-device-only -fsycl-targets=amdgcn-amd-amdhsa -Xsycl-target-backend --offload-arch=gfx906 -nogpulib %s -S -emit-llvm -o - | FileCheck %s --check-prefixes=CHECK

// This test checks that __usid_str has constant address space.

// CHECK: @__usid_str = private unnamed_addr addrspace(4) constant [38 x i8] c"uid

#include <sycl/sycl.hpp>

using namespace sycl;

constexpr specialization_id<unsigned int> SPEC_CONST(1024);

SYCL_EXTERNAL unsigned int foo(kernel_handler &kh) {
  return kh.get_specialization_constant<SPEC_CONST>();
}
