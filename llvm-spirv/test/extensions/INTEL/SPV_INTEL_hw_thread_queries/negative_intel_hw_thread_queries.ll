; Deprecation test for SPV_INTEL_hw_thread_queries extension.
; Check for errors in case if LLVM IR module contains HW thread queries as a for of
; SPIR-V friendly LLVM IR.
; RUN: llvm-as %s -o %t.bc
; RUN: not llvm-spirv %t.bc -o %t.spv 2>&1 | FileCheck %s

; CHECK: DeprecatedExtension: Feature requires the following deprecated SPIR-V extension:
; CHECK: SPV_INTEL_hw_thread_queries
; CHECK: Please report to https://github.com/intel/llvm in case if you see this error.
; CHECK: Ref LLVM Value:
; CHECK: @__spirv_BuiltInSubDeviceIDINTEL = external addrspace(1) global i32

target datalayout = "e-p:32:32-i64:64-v16:16-v24:32-v32:32-v48:64-v96:128-v192:256-v256:256-v512:512-v1024:1024"
target triple = "spir-unknown-unknown"

@__spirv_BuiltInSubDeviceIDINTEL = external addrspace(1) global i32

; Function Attrs: nounwind readnone
define spir_kernel void @foo() {
entry:
  %0 = load i32, ptr addrspace(4) addrspacecast (ptr addrspace(1) @__spirv_BuiltInSubDeviceIDINTEL to ptr addrspace(4)), align 4
  ret void
}
