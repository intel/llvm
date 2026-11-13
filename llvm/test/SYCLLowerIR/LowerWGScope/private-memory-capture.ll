; RUN: opt -passes=LowerWGScope,verify -S %s -o - | FileCheck %s

; Check the fixup of private_memory captures in the PFWI lambda object.
; The leader WI initializes the lambda object, which is then copied to all WIs.
; Each WI must then overwrite the captured private_memory addresses with the
; addresses of its own private_memory allocas, right before the PFWI call.
; Field 1 is captured in the generic address space, which requires an
; addrspacecast. Field 2 is captured in the private address space, which does
; not. Field 0 does not capture a private_memory variable and is not fixed up.

target datalayout = "e-i64:64-v16:16-v24:32-v32:32-v48:64-v96:128-v192:256-v256:256-v512:512-v1024:1024-n8:16:32:64-G1"
target triple = "spir64-unknown-unknown"

%lambda = type { ptr addrspace(4), ptr addrspace(4), ptr }

define internal spir_func void @pfwg(ptr addrspace(4) %this) !work_group_scope !0 {
entry:
  %pm.as4 = alloca i32, align 4, !work_item_scope !0
  %pm.as0 = alloca i32, align 4, !work_item_scope !0
  %obj = alloca %lambda, align 8
  %pm.as4.ascast = addrspacecast ptr %pm.as4 to ptr addrspace(4)
  %f0 = getelementptr inbounds %lambda, ptr %obj, i64 0, i32 0
  store ptr addrspace(4) %this, ptr %f0, align 8
  %f1 = getelementptr inbounds %lambda, ptr %obj, i64 0, i32 1
  store ptr addrspace(4) %pm.as4.ascast, ptr %f1, align 8
  %f2 = getelementptr inbounds %lambda, ptr %obj, i64 0, i32 2
  store ptr %pm.as0, ptr %f2, align 8
  call spir_func void @pfwi(ptr addrspace(4) %this, ptr %obj)
  ret void
}

; The original lambda object initialization is executed by the leader only.
; CHECK-LABEL: define internal spir_func void @pfwg(
; CHECK:       wg_leader:
; CHECK-NEXT:    %f0 = getelementptr inbounds %lambda, ptr %obj, i64 0, i32 0
; CHECK-NEXT:    store ptr addrspace(4) %this, ptr %f0, align 8
; CHECK-NEXT:    %f1 = getelementptr inbounds %lambda, ptr %obj, i64 0, i32 1
; CHECK-NEXT:    store ptr addrspace(4) %pm.as4.ascast, ptr %f1, align 8
; CHECK-NEXT:    %f2 = getelementptr inbounds %lambda, ptr %obj, i64 0, i32 2
; CHECK-NEXT:    store ptr %pm.as0, ptr %f2, align 8

; All WIs materialize the lambda object, then fix up the private_memory
; captures.
; CHECK:       LeaderMat:
; CHECK:         call void @llvm.memcpy.p0.p3.i64(ptr align 8 %obj, ptr addrspace(3) align 16 @WGCopy, i64 24, i1 false)
; CHECK-NEXT:    call spir_func void @_Z22__spirv_ControlBarrieriii(
; CHECK-NEXT:    [[F2:%.*]] = getelementptr inbounds %lambda, ptr %obj, i64 0, i32 2
; CHECK-NEXT:    store ptr %pm.as0, ptr [[F2]], align 8
; CHECK-NEXT:    [[F1:%.*]] = getelementptr inbounds %lambda, ptr %obj, i64 0, i32 1
; CHECK-NEXT:    [[CAST:%.*]] = addrspacecast ptr %pm.as4 to ptr addrspace(4)
; CHECK-NEXT:    store ptr addrspace(4) [[CAST]], ptr [[F1]], align 8
; CHECK-NEXT:    call spir_func void @pfwi(ptr addrspace(4) %this, ptr %obj)

define internal spir_func void @pfwi(ptr addrspace(4) %this, ptr %obj) !work_item_scope !0 !parallel_for_work_item !0 {
entry:
  ret void
}

!0 = !{}
