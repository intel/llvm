// RUN: %clang_cc1 -fsycl-is-device -triple spir64-unknown-unknown -disable-llvm-passes -emit-llvm -o - %s \
// RUN: | FileCheck %s --check-prefix=CHECK-DEVICE
// RUN: %clang_cc1 -fsycl-is-host   -triple x86_64-unknown-unknown -disable-llvm-passes -emit-llvm -o - %s -fcxx-exceptions\
// RUN: | FileCheck %s --check-prefix=CHECK-HOST

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
constexpr sycl::specialization_id<unsigned short> shortSize(1);

// COM: On the device, for each call, we should generate a chain of: 'call @sycl.alloca.<ty>' + ('addrspacecast') + 'store'.
// COM: The 'addrspacecast' will only appear when the pointer is not decorated, i.e., `DecorateAddress == sycl::access::decorated::no`.

// CHECK-DEVICE-LABEL: define dso_local spir_func void @_Z4testRN4sycl3_V114kernel_handlerE(
// CHECK-DEVICE-SAME: ptr addrspace(4) noundef align 1 dereferenceable(1) [[KH:%.*]])
// CHECK-DEVICE-NEXT:  entry:
// CHECK-DEVICE-NEXT:    [[KH_ADDR:%.*]] = alloca ptr addrspace(4), align 8
// CHECK-DEVICE-NEXT:    [[PTR0:%.*]] = alloca %"class.sycl::_V1::multi_ptr", align 8
// CHECK-DEVICE-NEXT:    [[TMP0:%.*]] = call ptr @llvm.sycl.alloca.p0.p4.p4.p4.f64(ptr addrspace(4) addrspacecast (ptr {{.*}} to ptr addrspace(4)), ptr addrspace(4) addrspacecast (ptr addrspace(1) {{.*}} to ptr addrspace(4)), ptr addrspace(4) null, double 0.000000e+00, i64 8)
// CHECK-DEVICE-NEXT:    [[PTR1:%.*]] = alloca %"class.sycl::_V1::multi_ptr.0", align 8
// CHECK-DEVICE-NEXT:    [[TMP2:%.*]] = call ptr @llvm.sycl.alloca.p0.p4.p4.p4.i32(ptr addrspace(4) addrspacecast (ptr {{.*}} to ptr addrspace(4)), ptr addrspace(4) addrspacecast (ptr addrspace(1) {{.*}} to ptr addrspace(4)), ptr addrspace(4) null, i32 0, i64 4)
// CHECK-DEVICE-NEXT:    [[PTR2:%.*]] = alloca %"class.sycl::_V1::multi_ptr.2", align 8
// CHECK-DEVICE-NEXT:    [[TMP4:%.*]] = call ptr @llvm.sycl.alloca.p0.p4.p4.p4.s_struct.myStructs(ptr addrspace(4) addrspacecast (ptr {{.*}} to ptr addrspace(4)), ptr addrspace(4) addrspacecast (ptr addrspace(1) {{.*}} to ptr addrspace(4)), ptr addrspace(4) null, %struct.myStruct zeroinitializer, i64 1)
// CHECK-DEVICE-NEXT:    [[KH_ADDR_ASCAST:%.*]] = addrspacecast ptr [[KH_ADDR]] to ptr addrspace(4)
// CHECK-DEVICE-NEXT:    [[PTR0_ASCAST:%.*]] = addrspacecast ptr [[PTR0]] to ptr addrspace(4)
// CHECK-DEVICE-NEXT:    [[PTR1_ASCAST:%.*]] = addrspacecast ptr [[PTR1]] to ptr addrspace(4)
// CHECK-DEVICE-NEXT:    [[PTR2_ASCAST:%.*]] = addrspacecast ptr [[PTR2]] to ptr addrspace(4)
// CHECK-DEVICE-NEXT:    [[TMP5:%.*]] = addrspacecast ptr [[TMP4]] to ptr addrspace(4)
// CHECK-DEVICE-NEXT:    store ptr addrspace(4) [[KH]], ptr addrspace(4) [[KH_ADDR_ASCAST]], align 8
// CHECK-DEVICE-NEXT:    store ptr [[TMP0]], ptr addrspace(4) [[PTR0_ASCAST]], align 8
// CHECK-DEVICE-NEXT:    store ptr [[TMP2]], ptr addrspace(4) [[PTR1_ASCAST]], align 8
// CHECK-DEVICE-NEXT:    store ptr addrspace(4) [[TMP5]], ptr addrspace(4) [[PTR2_ASCAST]], align 8
// CHECK-DEVICE-NEXT:    ret void

// COM: On the host, each call should be materialized...

// CHECK-HOST-LABEL: define dso_local void @_Z4testRN4sycl3_V114kernel_handlerE(
// CHECK-HOST-SAME: ptr noundef nonnull align 1 dereferenceable(1) [[KH:%.*]])
// CHECK-HOST-NEXT:  entry:
// CHECK-HOST-NEXT:    [[KH_ADDR:%.*]] = alloca ptr, align 8
// CHECK-HOST-NEXT:    [[PTR0:%.*]] = alloca %"class.sycl::_V1::multi_ptr", align 8
// CHECK-HOST-NEXT:    [[PTR1:%.*]] = alloca %"class.sycl::_V1::multi_ptr.0", align 8
// CHECK-HOST-NEXT:    [[PTR2:%.*]] = alloca %"class.sycl::_V1::multi_ptr.1", align 8
// CHECK-HOST-NEXT:    store ptr [[KH]], ptr [[KH_ADDR]], align 8
// CHECK-HOST-NEXT:    [[TMP0:%.*]] = load ptr, ptr [[KH_ADDR]], align 8
// CHECK-HOST-NEXT:    [[CALL:%.*]] = call ptr @_ZN4sycl3_V13ext6oneapi12experimental14private_allocaIdTnRDaL_ZL4sizeELNS0_6access9decoratedE1EEENS0_9multi_ptrIT_LNS6_13address_spaceE0EXT1_EEERNS0_14kernel_handlerE(ptr noundef nonnull align 1 dereferenceable(1) [[TMP0]])
// CHECK-HOST-NEXT:    [[COERCE_DIVE:%.*]] = getelementptr inbounds %"class.sycl::_V1::multi_ptr", ptr [[PTR0]], i32 0, i32 0
// CHECK-HOST-NEXT:    store ptr [[CALL]], ptr [[COERCE_DIVE]], align 8
// CHECK-HOST-NEXT:    [[TMP1:%.*]] = load ptr, ptr [[KH_ADDR]], align 8
// CHECK-HOST-NEXT:    [[CALL1:%.*]] = call ptr @_ZN4sycl3_V13ext6oneapi12experimental14private_allocaIiTnRDaL_ZL7intSizeELNS0_6access9decoratedE2EEENS0_9multi_ptrIT_LNS6_13address_spaceE0EXT1_EEERNS0_14kernel_handlerE(ptr noundef nonnull align 1 dereferenceable(1) [[TMP1]])
// CHECK-HOST-NEXT:    [[COERCE_DIVE2:%.*]] = getelementptr inbounds %"class.sycl::_V1::multi_ptr.0", ptr [[PTR1]], i32 0, i32 0
// CHECK-HOST-NEXT:    store ptr [[CALL1]], ptr [[COERCE_DIVE2]], align 8
// CHECK-HOST-NEXT:    [[TMP2:%.*]] = load ptr, ptr [[KH_ADDR]], align 8
// CHECK-HOST-NEXT:    [[CALL3:%.*]] = call ptr @_ZN4sycl3_V13ext6oneapi12experimental14private_allocaI8myStructTnRDaL_ZL7intSizeELNS0_6access9decoratedE0EEENS0_9multi_ptrIT_LNS7_13address_spaceE0EXT1_EEERNS0_14kernel_handlerE(ptr noundef nonnull align 1 dereferenceable(1) [[TMP2]])
// CHECK-HOST-NEXT:    [[COERCE_DIVE4:%.*]] = getelementptr inbounds %"class.sycl::_V1::multi_ptr.1", ptr [[PTR2]], i32 0, i32 0
// CHECK-HOST-NEXT:    store ptr [[CALL3]], ptr [[COERCE_DIVE4]], align 8
// CHECK-HOST-NEXT:    ret void
//
SYCL_EXTERNAL void test(sycl::kernel_handler &kh) {
  auto ptr0 = sycl::ext::oneapi::experimental::private_alloca<double, size, sycl::access::decorated::yes>(kh);
  auto ptr1 = sycl::ext::oneapi::experimental::private_alloca<int, intSize, sycl::access::decorated::legacy>(kh);
  auto ptr2 = sycl::ext::oneapi::experimental::private_alloca<myStruct, intSize, sycl::access::decorated::no>(kh);
}

// COM: And the body function should be simply a throw

// CHECK-HOST-LABEL: define internal ptr @_ZN4sycl3_V13ext6oneapi12experimental14private_allocaIdTnRDaL_ZL4sizeELNS0_6access9decoratedE1EEENS0_9multi_ptrIT_LNS6_13address_spaceE0EXT1_EEERNS0_14kernel_handlerE
// CHECK-HOST-SAME:  ptr noundef nonnull align 1 dereferenceable(1) [[H:%.*]])
// CHECK-HOST-NEXT:   entry:
// CHECK-HOST-NEXT:     [[H_ADDR:%.*]] = alloca ptr, align 8
// CHECK-HOST-NEXT:     store ptr [[H]], ptr [[H_ADDR]], align 8
// CHECK-HOST-NEXT:     [[EXCEPTION:%.*]] = call ptr @__cxa_allocate_exception(i64 8)
// CHECK-HOST-NEXT:     store ptr @.str, ptr [[EXCEPTION]], align 16
// CHECK-HOST-NEXT:     call void @__cxa_throw(ptr [[EXCEPTION]], ptr @_ZTIPKc, ptr null)
// CHECK-HOST-NEXT:     unreachable
