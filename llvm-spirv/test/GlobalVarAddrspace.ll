;
; This test case checks that LLVM -> SPIR-V translation produces valid
; SPIR-V module, where a global variable, defined with non-default
; address space, have correct non-function storage class.
;
; No additional checks are needed in addition to simple translation
; to SPIR-V. In case of an error newly produced SPIR-V module validation
; would fail due to spirv-val that detects problematic SPIR-V code from
; translator and reports it as the following error:
;
; "Variables can not have a function[7] storage class outside of a function".
;
;
; RUN: llvm-as %s -o %t.bc
; RUN: llvm-spirv %t.bc -o %t.spv
; RUN: spirv-val %t.spv

target datalayout = "e-i64:64-v16:16-v24:32-v32:32-v48:64-v96:128-v192:256-v256:256-v512:512-v1024:1024-n8:16:32:64"
target triple = "spir64-unknown-unknown"

@G = addrspace(1) global i1 true

define spir_func i1 @f(i1 %0) {
 store i1 %0, ptr addrspace(1) @G, align 1
 ret i1 %0
}
