;; Test to check that freeze instruction does not cause a crash
; RUN: llvm-spirv %s -o %t.spv
; RUN: llvm-spirv -r %t.spv -o %t.rev.bc
; All freeze instructions should be deleted and uses of freeze's result should be replaced
; with freeze's source or a random constant if freeze's source is poison or undef.
; RUN: llvm-dis < %t.rev.bc | FileCheck %s --check-prefix=CHECK-LLVM --implicit-check-not="= freeze"
; RUN: %if spirv-backend %{ llc -O0 -mtriple=spirv64-unknown-unknown -filetype=obj %s -o %t.llc.spv %}
; RUN: %if spirv-backend %{ llvm-spirv -r %t.llc.spv -o %t.llc.rev.bc %}
; RUN: %if spirv-backend %{ llvm-dis %t.llc.rev.bc -o %t.llc.rev.ll %}
; RUN: %if spirv-backend %{ FileCheck %s --check-prefix=CHECK-LLC --implicit-check-not="= freeze" < %t.llc.rev.ll %}

target datalayout = "e-i64:64-v16:16-v24:32-v32:32-v48:64-v96:128-v192:256-v256:256-v512:512-v1024:1024-n8:16:32:64"
target triple = "spir64-unknown-unknown"

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
; test i32
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;

; CHECK-LLVM: @testfunction_i32A
; Uses of result should be replaced with freeze's source
; CHECK-LLVM-NEXT: add nsw i32 %val, 1
define spir_func i32 @testfunction_i32A(i32 %val) {
   %1 = freeze i32 %val
   %2 = add nsw i32 %1, 1
   ret i32 %2
}

; CHECK-LLC: @testfunction_i32A
; CHECK-LLC-NEXT: add i32 %val, 1

; CHECK-LLVM: @testfunction_i32B
; Frozen poison/undef should produce a constant.
; add should be deleted since both inputs are constant.
; CHECK-LLVM-NEXT: ret i32
define spir_func i32 @testfunction_i32B(i32 %val) {
   %1 = freeze i32 poison
   %2 = add nsw i32 %1, 1
   ret i32 %2
}

; CHECK-LLC: @testfunction_i32B
; CHECK-LLC-NEXT: ret i32

; CHECK-LLVM: @testfunction_i32C
; Frozen poison/undef should produce a constant.
; add should be deleted since both inputs are constant.
; CHECK-LLVM-NEXT: ret i32
define spir_func i32 @testfunction_i32C(i32 %val) {
   %1 = freeze i32 undef
   %2 = add nsw i32 %1, 1
   ret i32 %2
}

; CHECK-LLC: @testfunction_i32C
; CHECK-LLC-NEXT: ret i32

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
; test float
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;

; CHECK-LLVM: @testfunction_floatA
; freeze should be eliminated.
; Uses of result should be replaced with freeze's source
; CHECK-LLVM-NEXT: fadd float %val
define spir_func float @testfunction_floatA(float %val) {
   %1 = freeze float %val
   %2 = fadd float %1, 1.0
   ret float %2
}

; CHECK-LLC: @testfunction_floatA
; CHECK-LLC-NEXT: fadd float %val

; CHECK-LLVM: @testfunction_floatB
; Frozen poison/undef should produce a constant.
; add should be deleted since both inputs are constant.
; CHECK-LLVM-NEXT: ret float
define spir_func float @testfunction_floatB(float %val) {
   %1 = freeze float poison
   %2 = fadd float %1, 1.0
   ret float %2
}

; CHECK-LLC: @testfunction_floatB
; CHECK-LLC-NEXT: ret float

; CHECK-LLVM: @testfunction_floatC
; Frozen poison/undef should produce a constant.
; add should be deleted since both inputs are constant.
; CHECK-LLVM-NEXT: ret float
define spir_func float @testfunction_floatC(float %val) {
   %1 = freeze float undef
   %2 = fadd float %1, 1.0
   ret float %2
}

; CHECK-LLC: @testfunction_floatC
; CHECK-LLC-NEXT: ret float

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
; test ptr
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;

; CHECK-LLVM: @testfunction_ptrA
; freeze should be eliminated.
; Uses of result should be replaced with freeze's source
; CHECK-LLVM-NEXT: ptrtoint ptr %val to i64
define spir_func i64 @testfunction_ptrA(ptr %val) {
   %1 = freeze ptr %val
   %2 = ptrtoint ptr %1 to i64
   ret i64 %2
}

; CHECK-LLC: @testfunction_ptrA
; CHECK-LLC-NEXT: ptrtoint ptr %val to i64

; CHECK-LLVM: @testfunction_ptrB
; Frozen poison/undef should produce a constant.
; For ptr type this constant is null.
; CHECK-LLVM-NEXT: ptrtoint ptr null to i64
define spir_func i64 @testfunction_ptrB(ptr addrspace(1) %val) {
   %1 = freeze ptr poison
   %2 = ptrtoint ptr %1 to i64
   ret i64 %2
}

; CHECK-LLC: @testfunction_ptrB
; CHECK-LLC-NEXT: ptrtoint ptr undef to i64

; CHECK-LLVM: @testfunction_ptrC
; Frozen poison/undef should produce a constant.
; For ptr type this constant is null.
; CHECK-LLVM-NEXT: ptrtoint ptr null to i64
define spir_func i64 @testfunction_ptrC(ptr addrspace(1) %val) {
   %1 = freeze ptr undef
   %2 = ptrtoint ptr %1 to i64
   ret i64 %2
}

; CHECK-LLC: @testfunction_ptrC
; CHECK-LLC-NEXT: ptrtoint ptr undef to i64

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
; test forward-referenced freeze (regression for "Id is not in map" assertion)
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
; A named value (%retval.1) is consumed by a freeze (%cond.fr) that is referenced by
; a loop-carried phi *before* it is defined.  When the freeze is moved away (no
; SPV_KHR_poison_freeze), its forward placeholder is resolved to the named %retval.1
; entry, re-keying that entry to the forward's id.  This used to leave the original
; id dangling in the OpName set and crash with "Id is not in map" at emission time.
; Now with explicit bookkeeping of NamedId in replaceForward, it shouldn't crash anymore.
; CHECK-LLVM: @testfunction_freeze_forward_ref
; CHECK-LLVM: %retval = phi i1 [ undef, %entry ], [ %cond.fr, %for.inc ]
; CHECK-LLVM: %cond.fr = or i1 %cmp14, %acc
define spir_func i1 @testfunction_freeze_forward_ref(ptr addrspace(4) %a, ptr addrspace(4) %b, i64 %n) {
entry:
  br label %for.cond

for.cond:
  %i       = phi i64 [ 0,         %entry   ], [ %i.inc,    %for.inc ]
  %retval  = phi i1  [ undef,     %entry   ], [ %cond.fr,  %for.inc ]
  %cmp5    = icmp slt i64 %i, %n
  br i1 %cmp5, label %for.body, label %exit

for.body:
  %pa      = getelementptr i8, ptr addrspace(4) %a, i64 %i
  %pb      = getelementptr i8, ptr addrspace(4) %b, i64 %i
  %va      = load i8, ptr addrspace(4) %pa, align 1
  %vb      = load i8, ptr addrspace(4) %pb, align 1
  %cmp14   = icmp ult i8 %va, %vb
  %cmp18   = icmp ule i8 %va, %vb
  %acc     = and i1 %cmp18, %retval
  %retval.1 = or i1 %cmp14, %acc
  %cond.fr = freeze i1 %retval.1
  %eq      = icmp eq i8 %va, %vb
  br i1 %eq, label %for.inc, label %exit

for.inc:
  %i.inc   = add nuw nsw i64 %i, 1
  br label %for.cond

exit:
  %r = phi i1 [ %retval, %for.cond ], [ %cond.fr, %for.body ]
  ret i1 %r
}
