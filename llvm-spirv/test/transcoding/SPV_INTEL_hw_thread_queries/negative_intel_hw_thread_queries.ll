; Negative test for SPV_INTEL_hw_thread_queries feature
; Check for errors in case if the extension is not enabled, but the appropriate
; SPV-IR patter is found
; RUN: llvm-as %s -o %t.bc
; RUN: not llvm-spirv %t.bc -o %t.spv 2>&1 | FileCheck %s

; CHECK: InvalidModule: Invalid SPIR-V module: Intel HW thread queries must be enabled by SPV_INTEL_hw_thread_queries extension.
; CHECK: LLVM value that is being translated:
; CHECK: @__spirv_BuiltInSubDeviceIDINTEL = external addrspace(1) global i32

target datalayout = "e-p:32:32-i64:64-v16:16-v24:32-v32:32-v48:64-v96:128-v192:256-v256:256-v512:512-v1024:1024"
target triple = "spir-unknown-unknown"

@__spirv_BuiltInSubDeviceIDINTEL = external addrspace(1) global i32

; Function Attrs: nounwind readnone
define spir_kernel void @foo() {
entry:
  %0 = load i32, i32 addrspace(4)* addrspacecast (i32 addrspace(1)* @__spirv_BuiltInSubDeviceIDINTEL to i32 addrspace(4)*), align 4
  ret void
}
