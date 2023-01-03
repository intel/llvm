; Added -opaque-pointers.
; FIXME: Align with the community code when project is ready to enable opaque
; pointers by default
; RUN: llvm-as < %s | llvm-dis -opaque-pointers | FileCheck %s
; Added -opaque-pointers.
; FIXME: Align with the community code when project is ready to enable opaque
; pointers by default
; RUN: verify-uselistorder -opaque-pointers %s

; Test that global aliases are allowed to be constant addrspacecast

@i = internal addrspace(1) global i8 42
@ia = internal alias ptr addrspace(2), addrspacecast (ptr addrspace(1) @i to ptr addrspace(3))
; CHECK: @ia = internal alias ptr addrspace(2), addrspacecast (ptr addrspace(1) @i to ptr addrspace(3))
