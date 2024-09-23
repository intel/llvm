; Ensure that a "common global" is not converted to a locally allocated
; variable when translated to SPIR-V and back to LLVM.

; RUN: llvm-as %s -o %t.bc
; RUN: llvm-spirv %t.bc -o %t.spv
; RUN: llvm-spirv -r %t.spv -o %t.rev.bc
; RUN: llvm-dis < %t.rev.bc | FileCheck %s --check-prefix=CHECK-LLVM

; CHECK-LLVM-NOT: alloca
; CHECK-LLVM: @CAG = common addrspace(1) global i32 0, align 4
; CHECK-LLVM-NOT: alloca

target triple = "spir64-unknown-unknown"

@CAG = common addrspace(1) global i32 0, align 4
  
define i32 @f() #0 {
 %1 = load i32, i32 addrspace(1) * @CAG, align 4
 ret i32 %1
}
