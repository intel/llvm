; This test checks to make sure that constant exprs fold in some simple situations

; Added -opaque-pointers.
; FIXME: Align with the community code when project is ready to enable opaque
; pointers by default
; RUN: llvm-as < %s | llvm-dis -opaque-pointers | FileCheck %s
; Added -opaque-pointers.
; FIXME: Align with the community code when project is ready to enable opaque
; pointers by default
; RUN: verify-uselistorder -opaque-pointers %s

; CHECK-NOT: bitcast
; CHECK-NOT: trunc
; CHECK: addrspacecast
; CHECK: addrspacecast

@A = global ptr null  ; Cast null -> fold
@B = global ptr @A   ; Cast to same type -> fold
@C = global i32 trunc (i64 42 to i32)        ; Integral casts
@D = global ptr @C  ; cast of cast ptr->ptr
@E = global i32 ptrtoint(ptr inttoptr (i8 5 to ptr) to i32)  ; i32 -> ptr -> i32

; Test folding of binary instrs
@F = global ptr inttoptr (i32 add (i32 5, i32 -5) to ptr)
@G = global ptr inttoptr (i32 sub (i32 5, i32 5) to ptr)

; Address space cast AS0 null-> AS1 null
@H = global ptr addrspace(1) addrspacecast(ptr null to ptr addrspace(1))

; Address space cast AS1 null-> AS0 null
@I = global ptr addrspacecast(ptr addrspace(1) null to ptr)

; Bitcast -> GEP
@J = external global { i32 }
@K = global ptr @J
