; RUN: not --crash opt -passes=LowerWGScope -disable-output %s 2>&1 | FileCheck %s

; Check that LowerWGScope reports an error when the address of a private_memory
; capture field in the PFWI lambda object depends on a value other than the
; lambda object itself or a constant. The field address is cloned right before
; the PFWI call, where such a value may not be available.

; CHECK: LLVM ERROR: LowerWGScope: private_memory capture field address has operands that may not dominate the PFWI call

target datalayout = "e-i64:64-v16:16-v24:32-v32:32-v48:64-v96:128-v192:256-v256:256-v512:512-v1024:1024-n8:16:32:64-G1"
target triple = "spir64-unknown-unknown"

%lambda = type { [2 x ptr addrspace(4)] }

define internal spir_func void @pfwg(ptr addrspace(4) %this, i64 %n) !work_group_scope !0 {
entry:
  %pm = alloca i32, align 4, !work_item_scope !0
  %obj = alloca %lambda, align 8
  %pm.ascast = addrspacecast ptr %pm to ptr addrspace(4)
  %idx = and i64 %n, 1
  %f = getelementptr inbounds %lambda, ptr %obj, i64 0, i32 0, i64 %idx
  store ptr addrspace(4) %pm.ascast, ptr %f, align 8
  call spir_func void @pfwi(ptr addrspace(4) %this, ptr %obj)
  ret void
}

define internal spir_func void @pfwi(ptr addrspace(4) %this, ptr %obj) !work_item_scope !0 !parallel_for_work_item !0 {
entry:
  ret void
}

!0 = !{}
