; RUN: opt < %s -passes=sycllowerwglocalmemory -S | FileCheck %s

; `foo` is a SYCL_EXTERNAL function that directly calls `group_local_memory_for_overwrite`.
; Frontend propagates `sycl-forceinline` attribute from `group_local_memory_for_overwrite` to `foo`.
; This test checks that `foo` is not inlined.

target datalayout = "e-i64:64-v16:16-v24:32-v32:32-v48:64-v96:128-v192:256-v256:256-v512:512-v1024:1024-n8:16:32:64-G1"
target triple = "spir64-unknown-unknown"

%"class.sycl::_V1::multi_ptr" = type { ptr addrspace(3) }
%"class.sycl::_V1::group" = type { %"class.sycl::_V1::range", %"class.sycl::_V1::range", %"class.sycl::_V1::range", %"class.sycl::_V1::id" }
%"class.sycl::_V1::range" = type { %"class.sycl::_V1::detail::array" }
%"class.sycl::_V1::detail::array" = type { [3 x i64] }
%"class.sycl::_V1::id" = type { %"class.sycl::_V1::detail::array" }

; CHECK: @WGLocalMem = internal addrspace(3) global [0 x i8] poison, align 1

define weak_odr dso_local spir_func void @_Z3fooPPi(ptr addrspace(4) noundef %a) #0 {
entry:
; CHECK-LABEL: define {{.*}}  @_Z3fooPPi(
; CHECK: store ptr addrspace(3) @WGLocalMem,

  call spir_func void @_ZN4sycl3_V13ext6oneapi32group_local_memory_for_overwriteIiNS0_5groupILi3EEEEENSt9enable_ifIXaasr3stdE27is_trivially_destructible_vIT_Esr4sycl6detail8is_groupIT0_EE5valueENS0_9multi_ptrIS7_LNS0_6access13address_spaceE3ELNSA_9decoratedE2EEEE4typeES8_(ptr addrspace(4) null, ptr null)
  ret void
}

define linkonce_odr dso_local spir_func void @_ZN4sycl3_V13ext6oneapi32group_local_memory_for_overwriteIiNS0_5groupILi3EEEEENSt9enable_ifIXaasr3stdE27is_trivially_destructible_vIT_Esr4sycl6detail8is_groupIT0_EE5valueENS0_9multi_ptrIS7_LNS0_6access13address_spaceE3ELNSA_9decoratedE2EEEE4typeES8_(ptr addrspace(4) sret(%"class.sycl::_V1::multi_ptr") align 8 %result, ptr noundef byval(%"class.sycl::_V1::group") align 8 %g) #1 {
entry:
; CHECK-LABEL: define {{.*}} @_ZN4sycl3_V13ext6oneapi32group_local_memory_for_overwriteIiNS0_5groupILi3EEEEENSt9enable_ifIXaasr3stdE27is_trivially_destructible_vIT_Esr4sycl6detail8is_groupIT0_EE5valueENS0_9multi_ptrIS7_LNS0_6access13address_spaceE3ELNSA_9decoratedE2EEEE4typeES8_(

  %AllocatedMem.ascast = addrspacecast ptr %g to ptr addrspace(4)
  %call = call spir_func ptr addrspace(3) @__sycl_allocateLocalMemory(i64 0, i64 1)
  store ptr addrspace(3) %call, ptr addrspace(4) %AllocatedMem.ascast, align 8
  ret void
}

declare spir_func ptr addrspace(3) @__sycl_allocateLocalMemory(i64, i64)

define internal spir_func void @_ZZZ4mainENKUlRN4sycl3_V17handlerEE_clES2_ENKUlNS0_7nd_itemILi1EEEE_clES5_() {
entry:
; CHECK-LABEL: define {{.*}} @_ZZZ4mainENKUlRN4sycl3_V17handlerEE_clES2_ENKUlNS0_7nd_itemILi1EEEE_clES5_(
; CHECK: call spir_func void @_Z3fooPPi(

  call spir_func void @_Z3fooPPi(ptr addrspace(4) null)
  ret void
}

attributes #0 = { "sycl-forceinline"="true" "sycl-module-id"="group_local_memory_template.cpp" }
attributes #1 = { "sycl-forceinline"="true" }
