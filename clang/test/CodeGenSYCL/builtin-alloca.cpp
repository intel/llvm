// RUN: %clang_cc1 -fsycl-is-device -triple spir64-unknown-unknown -disable-llvm-passes -emit-llvm -o - %s \
// RUN: | FileCheck %s

// Test codegen for __builtin_intel_sycl_alloca.

#include <stddef.h>

#include "Inputs/sycl.hpp"
#include "Inputs/private_alloca.hpp"

// expected-no-diagnostics

struct myStruct {
  char a;
  char b;
};

constexpr sycl::specialization_id<size_t> size(1);
constexpr sycl::specialization_id<int> intSize(-1);

// For each call, we should generate a chain of: 'call @llvm.sycl.alloca.<ty>' + ('addrspacecast') + 'store'.
// The 'addrspacecast' will only appear when the pointer is not decorated, i.e., `DecorateAddress == sycl::access::decorated::no`.

// CHECK-LABEL: define dso_local spir_func void @_Z4testRN4sycl3_V114kernel_handlerE(
// CHECK-SAME: ptr addrspace(4) noundef align 1 dereferenceable(1) [[KH:%.*]])
// CHECK-NEXT:  entry:
// CHECK-NEXT:    [[KH_ADDR:%.*]] = alloca ptr addrspace(4), align 8
// CHECK-NEXT:    [[PTR0:%.*]] = alloca %"class.sycl::_V1::multi_ptr", align 8
// CHECK-NEXT:    [[TMP0:%.*]] = call ptr @llvm.sycl.alloca.p0.p4.p4.p4.f64(ptr addrspace(4) addrspacecast (ptr {{.*}} to ptr addrspace(4)), ptr addrspace(4) addrspacecast (ptr addrspace(1) {{.*}} to ptr addrspace(4)), ptr addrspace(4) null, double 0.000000e+00, i64 8)
// CHECK-NEXT:    [[PTR1:%.*]] = alloca %"class.sycl::_V1::multi_ptr.0", align 8
// CHECK-NEXT:    [[TMP2:%.*]] = call ptr @llvm.sycl.alloca.p0.p4.p4.p4.i32(ptr addrspace(4) addrspacecast (ptr {{.*}} to ptr addrspace(4)), ptr addrspace(4) addrspacecast (ptr addrspace(1) {{.*}} to ptr addrspace(4)), ptr addrspace(4) null, i32 0, i64 4)
// CHECK-NEXT:    [[PTR2:%.*]] = alloca %"class.sycl::_V1::multi_ptr.2", align 8
// CHECK-NEXT:    [[TMP4:%.*]] = call ptr @llvm.sycl.alloca.p0.p4.p4.p4.s_struct.myStructs(ptr addrspace(4) addrspacecast (ptr {{.*}} to ptr addrspace(4)), ptr addrspace(4) addrspacecast (ptr addrspace(1) {{.*}} to ptr addrspace(4)), ptr addrspace(4) null, %struct.myStruct zeroinitializer, i64 1)
// CHECK-NEXT:    [[KH_ADDR_ASCAST:%.*]] = addrspacecast ptr [[KH_ADDR]] to ptr addrspace(4)
// CHECK-NEXT:    [[PTR0_ASCAST:%.*]] = addrspacecast ptr [[PTR0]] to ptr addrspace(4)
// CHECK-NEXT:    [[PTR1_ASCAST:%.*]] = addrspacecast ptr [[PTR1]] to ptr addrspace(4)
// CHECK-NEXT:    [[PTR2_ASCAST:%.*]] = addrspacecast ptr [[PTR2]] to ptr addrspace(4)
// CHECK-NEXT:    [[TMP5:%.*]] = addrspacecast ptr [[TMP4]] to ptr addrspace(4)
// CHECK-NEXT:    store ptr addrspace(4) [[KH]], ptr addrspace(4) [[KH_ADDR_ASCAST]], align 8
// CHECK-NEXT:    store ptr [[TMP0]], ptr addrspace(4) [[PTR0_ASCAST]], align 8
// CHECK-NEXT:    store ptr [[TMP2]], ptr addrspace(4) [[PTR1_ASCAST]], align 8
// CHECK-NEXT:    store ptr addrspace(4) [[TMP5]], ptr addrspace(4) [[PTR2_ASCAST]], align 8
// CHECK-NEXT:    ret void
SYCL_EXTERNAL void test(sycl::kernel_handler &kh) {
  auto ptr0 = sycl::ext::oneapi::experimental::private_alloca<double, size, sycl::access::decorated::yes>(kh);
  auto ptr1 = sycl::ext::oneapi::experimental::private_alloca<int, intSize, sycl::access::decorated::legacy>(kh);
  auto ptr2 = sycl::ext::oneapi::experimental::private_alloca<myStruct, intSize, sycl::access::decorated::no>(kh);
}
