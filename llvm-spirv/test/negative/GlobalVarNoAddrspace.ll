; Ensure "global" with no addrspace is reported as an error
; since the SPIR-V spec states:
;
;   global variable declarations must always have an address space
;   specified and that address space cannot be `0`

; RUN: llvm-as %s -o %t.bc
; RUN: not llvm-spirv %t.bc 2>&1 \
; RUN: | FileCheck %s --check-prefix=CHECK-ERROR

; CHECK-ERROR: InvalidInstruction: Can't translate llvm instruction:
; CHECK-ERROR-NEXT: Global variable cannot have Function storage class. Consider setting a proper address space.
; CHECK-ERROR-NEXT: Original LLVM value:
; CHECK-ERROR-NEXT: @G = global i1 true

target datalayout = "e-i64:64-v16:16-v24:32-v32:32-v48:64-v96:128-v192:256-v256:256-v512:512-v1024:1024-n8:16:32:64"
target triple = "spir64-unknown-unknown"

@G = global i1 true

define spir_func i1 @f(i1 %0) {
 store i1 %0, ptr @G, align 1
 ret i1 %0
}
