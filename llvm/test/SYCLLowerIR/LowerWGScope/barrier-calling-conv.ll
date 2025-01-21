; RUN: opt -passes=LowerWGScope -S %s -o - | FileCheck %s

; Check newly created barrier call has spir_func calling convention.

target datalayout = "e-i64:64-v16:16-v24:32-v32:32-v48:64-v96:128-v192:256-v256:256-v512:512-v1024:1024-n8:16:32:64-G1"
target triple = "spir64-unknown-unknown"

%"class.sycl::_V1::group" = type { %"class.sycl::_V1::range", %"class.sycl::_V1::range", %"class.sycl::_V1::range", %"class.sycl::_V1::id" }
%"class.sycl::_V1::range" = type { %"class.sycl::_V1::detail::array" }
%"class.sycl::_V1::detail::array" = type { [1 x i64] }
%"class.sycl::_V1::id" = type { %"class.sycl::_V1::detail::array" }

define internal spir_func void @_ZZZ4mainENKUlRN4sycl3_V17handlerEE_clES2_ENKUlNS0_5groupILi1EEEE_clES5_(ptr addrspace(4) noundef align 1 dereferenceable_or_null(1) %this, ptr noundef byval(%"class.sycl::_V1::group") align 8 %group) !work_group_scope !0 {
entry:
; CHECK: call spir_func void @_Z22__spirv_ControlBarrierjjj(

  %this.addr = alloca ptr addrspace(4), align 8
  %this.addr.ascast = addrspacecast ptr %this.addr to ptr addrspace(4)
  store ptr addrspace(4) %this, ptr addrspace(4) %this.addr.ascast, align 8
  %group.ascast = addrspacecast ptr %group to ptr addrspace(4)
  %this1 = load ptr addrspace(4), ptr addrspace(4) %this.addr.ascast, align 8
  ret void
}

; CHECK: declare spir_func void @_Z22__spirv_ControlBarrierjjj(

!0 = !{}
