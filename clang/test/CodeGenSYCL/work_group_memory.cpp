// RUN: %clang_cc1 -fsycl-is-device -triple spir64-unknown-unknown -disable-llvm-passes -emit-llvm %s -o %t.ll
// RUN: FileCheck < %t.ll %s --check-prefix CHECK-IR
// RUN: %clang_cc1 -fsycl-is-device -triple spir64-unknown-unknown -fsycl-int-header=%t.h %s
// RUN: FileCheck < %t.h %s --check-prefix CHECK-INT-HEADER
//
// Tests for work_group_memory kernel parameter using the dummy implementation in Inputs/sycl.hpp.
// The first two RUN commands verify that the init call is generated with the correct arguments in LLVM IR
// and the second two RUN commands verify the contents of the integration header produced by the frontend.
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
// CHECK-INT-HEADER-NEXT: //--- _ZTSZZ4mainENKUlRN4sycl3_V17handlerEE_clES2_EUlNS0_4itemILi1EEEE_
// CHECK-INT-HEADER-NEXT: { kernel_param_kind_t::kind_work_group_memory, {{[4,8]}}, 0 },
// CHECK-INT-HEADER-EMPTY:
// CHECK-INT-HEADER-NEXT: { kernel_param_kind_t::kind_invalid, -987654321, -987654321 },
// CHECK-INT-HEADER-NEXT: };

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
