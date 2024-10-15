// RUN: %clang_cc1 -fsycl-is-device -triple spir64-unknown-unknown -disable-llvm-passes -emit-llvm %s -o %t.ll
// RUN: FileCheck < %t.ll %s --check-prefix CHECK-IR
// RUN: %clang_cc1 -fsycl-is-device -triple spir64-unknown-unknown -fsycl-int-header=%t.h %s
// RUN: FileCheck < %t.h %s --check-prefix CHECK-INT-HEADER
//
// CHECK-IR: define dso_local spir_kernel void @
// CHECK-IR-SAME: ptr addrspace(3) noundef align 4 [[PTR:%[a-zA-Z0-9_]+]]
//
// CHECK-IR: [[PTR]].addr = alloca ptr addrspace(3), align 8
// CHECK-IR: [[PTR]].addr.ascast = addrspacecast ptr [[PTR]].addr to ptr addrspace(4)
// CHECK-IR: store ptr addrspace(3) [[PTR]], ptr addrspace(4) [[PTR]].addr.ascast, align 8
// CHECK-IR: [[PTR_LOAD:%[a-zA-Z0-9_]+]] = load ptr addrspace(3), ptr addrspace(4) [[PTR]].addr.ascast, align 8
//
// CHECK-IR: call spir_func void @{{.*}}__init{{.*}}(ptr addrspace(4) noundef align 8 dereferenceable_or_null(8) %{{[a-zA-Z0-9_]+}}, ptr addrspace(3) noundef [[PTR_LOAD]])
//
// CHECK-INT-HEADER: const kernel_param_desc_t kernel_signatures[] = {
// CHECK-INT-HEADER-NEXT: //{{.*}}
// CHECK-INT-HEADER-NEXT: { kernel_param_kind_t::kind_work_group_memory, 8, 0 },

#include "Inputs/sycl.hpp"

int main() {
  sycl::queue Q;
  Q.submit([&](sycl::handler &CGH) {
    sycl::work_group_memory<int> mem;
    sycl::range<1> ndr;
    CGH.parallel_for(ndr, [=](sycl::item<1> it) { int *ptr = &mem; });
  });
  return 0;
}
