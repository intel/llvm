; RUN: opt < %s -passes=sycllowerwglocalmemory -S | FileCheck %s

; Check group_local_memory_for_overwrite and group_local_memory functions are inlined.
; Check __sycl_allocateLocalMemory calls are lowered to four separate allocations.

target datalayout = "e-i64:64-v16:16-v24:32-v32:32-v48:64-v96:128-v192:256-v256:256-v512:512-v1024:1024-n8:16:32:64-G1"
target triple = "spir64-unknown-unknown"

%"class.sycl::_V1::multi_ptr" = type { ptr addrspace(3) }
%"class.sycl::_V1::group" = type { %"class.sycl::_V1::range", %"class.sycl::_V1::range", %"class.sycl::_V1::range", %"class.sycl::_V1::id" }
%"class.sycl::_V1::range" = type { %"class.sycl::_V1::detail::array" }
%"class.sycl::_V1::detail::array" = type { [1 x i64] }
%"class.sycl::_V1::id" = type { %"class.sycl::_V1::detail::array" }

; CHECK: @WGLocalMem{{.*}} = internal addrspace(3) global [4 x i8] undef, align 4
; CHECK: @WGLocalMem{{.*}} = internal addrspace(3) global [4 x i8] undef, align 4
; CHECK: @WGLocalMem{{.*}} = internal addrspace(3) global [4 x i8] undef, align 4
; CHECK: @WGLocalMem{{.*}} = internal addrspace(3) global [4 x i8] undef, align 4

; Function Attrs: alwaysinline
define internal spir_func void @_ZZZ4mainENKUlRN4sycl3_V17handlerEE_clES2_ENKUlNS0_7nd_itemILi1EEEE_clES5_() #0 {
entry:
; CHECK: define internal spir_func void @_ZZZ4mainENKUlRN4sycl3_V17handlerEE_clES2_ENKUlNS0_7nd_itemILi1EEEE_clES5_(
; CHECK: store ptr addrspace(3) @WGLocalMem{{.*}}, ptr addrspace(4) %AllocatedMem{{.*}}, align 8
; CHECK: store ptr addrspace(3) @WGLocalMem{{.*}}, ptr addrspace(4) %AllocatedMem{{.*}}, align 8
; CHECK: store ptr addrspace(3) @WGLocalMem{{.*}}, ptr addrspace(4) %AllocatedMem{{.*}}, align 8
; CHECK: store ptr addrspace(3) @WGLocalMem{{.*}}, ptr addrspace(4) %AllocatedMem{{.*}}, align 8

  %Ptr = alloca %"class.sycl::_V1::multi_ptr", align 8
  %agg = alloca %"class.sycl::_V1::group", align 8
  %Ptr.ascast = addrspacecast ptr %Ptr to ptr addrspace(4)
  call spir_func void @_ZN4sycl3_V13ext6oneapi32group_local_memory_for_overwriteIiNS0_5groupILi1EEEEENSt9enable_ifIXaasr3stdE27is_trivially_destructible_vIT_Esr4sycl6detail8is_groupIT0_EE5valueENS0_9multi_ptrIS7_LNS0_6access13address_spaceE3ELNSA_9decoratedE2EEEE4typeES8_(ptr addrspace(4) sret(%"class.sycl::_V1::multi_ptr") align 8 %Ptr.ascast, ptr byval(%"class.sycl::_V1::group") align 8 %agg)
  call spir_func void @_ZN4sycl3_V13ext6oneapi32group_local_memory_for_overwriteIiNS0_5groupILi1EEEEENSt9enable_ifIXaasr3stdE27is_trivially_destructible_vIT_Esr4sycl6detail8is_groupIT0_EE5valueENS0_9multi_ptrIS7_LNS0_6access13address_spaceE3ELNSA_9decoratedE2EEEE4typeES8_(ptr addrspace(4) sret(%"class.sycl::_V1::multi_ptr") align 8 %Ptr.ascast, ptr byval(%"class.sycl::_V1::group") align 8 %agg)
  call spir_func void @_ZN4sycl3_V13ext6oneapi18group_local_memoryIiNS0_5groupILi1EEEJEEENSt9enable_ifIXaasr3stdE27is_trivially_destructible_vIT_Esr4sycl6detail8is_groupIT0_EE5valueENS0_9multi_ptrIS7_LNS0_6access13address_spaceE3ELNSA_9decoratedE2EEEE4typeES8_DpOT1_(ptr addrspace(4) sret(%"class.sycl::_V1::multi_ptr") align 8 %Ptr.ascast, ptr byval(%"class.sycl::_V1::group") align 8 %agg)
  call spir_func void @_ZN4sycl3_V13ext6oneapi18group_local_memoryIiNS0_5groupILi1EEEJEEENSt9enable_ifIXaasr3stdE27is_trivially_destructible_vIT_Esr4sycl6detail8is_groupIT0_EE5valueENS0_9multi_ptrIS7_LNS0_6access13address_spaceE3ELNSA_9decoratedE2EEEE4typeES8_DpOT1_(ptr addrspace(4) sret(%"class.sycl::_V1::multi_ptr") align 8 %Ptr.ascast, ptr byval(%"class.sycl::_V1::group") align 8 %agg)
  ret void
}

; CHECK-NOT: define {{.*}} @_ZN4sycl3_V13ext6oneapi32group_local_memory_for_overwriteIiNS0_5groupILi1EEEEENSt9enable_ifIXaasr3stdE27is_trivially_destructible_vIT_Esr4sycl6detail8is_groupIT0_EE5valueENS0_9multi_ptrIS7_LNS0_6access13address_spaceE3ELNSA_9decoratedE2EEEE4typeES8_(

; Function Attrs: alwaysinline
define spir_func void @_ZN4sycl3_V13ext6oneapi32group_local_memory_for_overwriteIiNS0_5groupILi1EEEEENSt9enable_ifIXaasr3stdE27is_trivially_destructible_vIT_Esr4sycl6detail8is_groupIT0_EE5valueENS0_9multi_ptrIS7_LNS0_6access13address_spaceE3ELNSA_9decoratedE2EEEE4typeES8_(ptr addrspace(4) sret(%"class.sycl::_V1::multi_ptr") align 8 %agg.result, ptr byval(%"class.sycl::_V1::group") align 8 %g) #0 {
entry:
  %AllocatedMem = alloca ptr addrspace(3), align 8
  %AllocatedMem.ascast = addrspacecast ptr %AllocatedMem to ptr addrspace(4)
  %call = call spir_func ptr addrspace(3) @__sycl_allocateLocalMemory(i64 4, i64 4)
  store ptr addrspace(3) %call, ptr addrspace(4) %AllocatedMem.ascast, align 8
  ret void
}

; CHECK-NOT: define {{.*}} @_ZN4sycl3_V13ext6oneapi18group_local_memoryIiNS0_5groupILi1EEEJEEENSt9enable_ifIXaasr3stdE27is_trivially_destructible_vIT_Esr4sycl6detail8is_groupIT0_EE5valueENS0_9multi_ptrIS7_LNS0_6access13address_spaceE3ELNSA_9decoratedE2EEEE4typeES8_DpOT1_(

; Function Attrs: alwaysinline
define spir_func void @_ZN4sycl3_V13ext6oneapi18group_local_memoryIiNS0_5groupILi1EEEJEEENSt9enable_ifIXaasr3stdE27is_trivially_destructible_vIT_Esr4sycl6detail8is_groupIT0_EE5valueENS0_9multi_ptrIS7_LNS0_6access13address_spaceE3ELNSA_9decoratedE2EEEE4typeES8_DpOT1_(ptr addrspace(4) sret(%"class.sycl::_V1::multi_ptr") align 8 %agg.result, ptr byval(%"class.sycl::_V1::group") align 8 %g) #0 {
entry:
  %AllocatedMem = alloca ptr addrspace(3), align 8
  %AllocatedMem.ascast = addrspacecast ptr %AllocatedMem to ptr addrspace(4)
  %call = call spir_func ptr addrspace(3) @__sycl_allocateLocalMemory(i64 4, i64 4)
  store ptr addrspace(3) %call, ptr addrspace(4) %AllocatedMem.ascast, align 8
  ret void
}

declare spir_func ptr addrspace(3) @__sycl_allocateLocalMemory(i64 noundef, i64 noundef)

attributes #0 = { alwaysinline }
