; RUN: opt -passes=LowerWGScope,verify -S %s -o - | FileCheck %s

; Check the fixup of a private_memory capture in the PFWI lambda object when the
; capture field address is a byte-offset GEP (ptradd form), which carries no
; information about the lambda object type.

target datalayout = "e-i64:64-v16:16-v24:32-v32:32-v48:64-v96:128-v192:256-v256:256-v512:512-v1024:1024-n8:16:32:64-G1"
target triple = "spir64-unknown-unknown"

%lambda = type { ptr addrspace(4), ptr addrspace(4) }

define internal spir_func void @pfwg(ptr addrspace(4) %this) !work_group_scope !0 {
entry:
  %pm = alloca i32, align 4, !work_item_scope !0
  %obj = alloca %lambda, align 8
  %pm.ascast = addrspacecast ptr %pm to ptr addrspace(4)
  %f1 = getelementptr inbounds i8, ptr %obj, i64 8
  store ptr addrspace(4) %pm.ascast, ptr %f1, align 8
  call spir_func void @pfwi(ptr addrspace(4) %this, ptr %obj)
  ret void
}

; The original capture initialization is executed by the leader only.
; CHECK-LABEL: define internal spir_func void @pfwg(
; CHECK:       wg_leader:
; CHECK-NEXT:    %f1 = getelementptr inbounds i8, ptr %obj, i64 8
; CHECK-NEXT:    store ptr addrspace(4) %pm.ascast, ptr %f1, align 8

; All WIs materialize the lambda object, then fix up the capture.
; CHECK:       LeaderMat:
; CHECK:         call void @llvm.memcpy.p0.p3.i64(ptr align 8 %obj, ptr addrspace(3) align {{[0-9]+}} @WGCopy, i64 16, i1 false)
; CHECK-NEXT:    call spir_func void @_Z22__spirv_ControlBarrieriii(
; CHECK-NEXT:    [[F1:%.*]] = getelementptr inbounds i8, ptr %obj, i64 8
; CHECK-NEXT:    [[CAST:%.*]] = addrspacecast ptr %pm to ptr addrspace(4)
; CHECK-NEXT:    store ptr addrspace(4) [[CAST]], ptr [[F1]], align 8
; CHECK-NEXT:    call spir_func void @pfwi(ptr addrspace(4) %this, ptr %obj)

define internal spir_func void @pfwi(ptr addrspace(4) %this, ptr %obj) !work_item_scope !0 !parallel_for_work_item !0 {
entry:
  ret void
}

!0 = !{}
