; RUN: opt -passes=LowerWGScope -S %s -o - | FileCheck %s

; Check that no illegal AS casts remain after the pass

target datalayout = "e-i64:64-v16:16-v24:32-v32:32-v48:64-v96:128-v192:256-v256:256-v512:512-v1024:1024-n8:16:32:64-G1"
target triple = "spir64-unknown-unknown"

%"class.sycl::_V1::group" = type { %"class.sycl::_V1::range", %"class.sycl::_V1::range", %"class.sycl::_V1::range", %"class.sycl::_V1::id" }
%"class.sycl::_V1::range" = type { %"class.sycl::_V1::detail::array" }
%"class.sycl::_V1::detail::array" = type { [1 x i64] }
%"class.sycl::_V1::id" = type { %"class.sycl::_V1::detail::array" }

@hierarchical = internal addrspace(3) global %"class.sycl::_V1::range" undef, align 8

define internal spir_func void @foo(ptr addrspace(4) noundef align 1 dereferenceable_or_null(1) %this, ptr noundef byval(%"class.sycl::_V1::group") align 8 %group_pid) !work_group_scope !0 {
entry:
; CHECK: entry:
; CHECK-NEXT: %lower_wg.local_copy = alloca %"class.sycl::_V1::id", align 8
; CHECK: call spir_func void @_ZNK4sycl3_V15groupILi1EE12get_group_idEv(ptr dead_on_unwind writable sret(%"class.sycl::_V1::id") align 8 %lower_wg.local_copy, ptr {{.*}})
; CHECK: %lower_wg.private_load = load %"class.sycl::_V1::id", ptr %lower_wg.local_copy, align 8
; CHECK: store %"class.sycl::_V1::id" %lower_wg.private_load, ptr addrspace(3) @hierarchical, align 8
; CHECK-NOT: addrspacecast (ptr addrspace(3) @hierarchical to ptr)
  %group_pid.ascast = addrspacecast ptr %group_pid to ptr addrspace(4)
  call spir_func void @_ZNK4sycl3_V15groupILi1EE12get_group_idEv(ptr dead_on_unwind writable sret(%"class.sycl::_V1::id") align 8 addrspacecast (ptr addrspace(3) @hierarchical to ptr), ptr addrspace(4) noundef align 8 dereferenceable_or_null(32) %group_pid.ascast)
  ret void
}

declare spir_func void @_ZNK4sycl3_V15groupILi1EE12get_group_idEv(ptr dead_on_unwind noalias writable sret(%"class.sycl::_V1::id") align 8 %agg.result, ptr addrspace(4) noundef align 8 dereferenceable_or_null(32) %this)

!0 = !{}
