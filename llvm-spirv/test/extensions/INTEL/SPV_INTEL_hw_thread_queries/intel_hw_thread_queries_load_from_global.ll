; Test for SPV_INTEL_hw_thread_queries feature
; SPIR-V friendly LLVM IR to SPIR-V translation and vice versa
; RUN: llvm-as %s -o %t.bc
; RUN: llvm-spirv %t.bc --spirv-ext=+SPV_INTEL_hw_thread_queries -o %t.spv
; RUN: llvm-spirv %t.spv -to-text -o %t.spt
; RUN: FileCheck < %t.spt %s --check-prefix=CHECK-SPIRV

; RUN: llvm-spirv -r %t.spv -o %t.rev.bc
; RUN: llvm-dis < %t.rev.bc | FileCheck %s --check-prefix=CHECK-LLVM
; RUN: llvm-spirv -r --spirv-target-env=SPV-IR %t.spv -o %t.rev.bc
; RUN: llvm-dis < %t.rev.bc | FileCheck %s --check-prefix=CHECK-LLVM

; CHECK-SPIRV: Capability HWThreadQueryINTEL
; CHECK-SPIRV: Extension "SPV_INTEL_hw_thread_queries"
; CHECK-SPIRV: Decorate [[#Id1:]] BuiltIn 6135
; CHECK-SPIRV: Decorate [[#Id2:]] BuiltIn 6136
; CHECK-SPIRV: Variable [[#]] [[#Id1]]
; CHECK-SPIRV: Variable [[#]] [[#Id2]]

target datalayout = "e-p:32:32-i64:64-v16:16-v24:32-v32:32-v48:64-v96:128-v192:256-v256:256-v512:512-v1024:1024"
target triple = "spir-unknown-unknown"

@__spirv_BuiltInSubDeviceIDINTEL = external addrspace(1) global i32
@__spirv_BuiltInGlobalHWThreadIDINTEL = external addrspace(1) global i32

; Function Attrs: nounwind readnone
define spir_kernel void @foo() {
entry:
  %0 = load i32, i32 addrspace(4)* addrspacecast (i32 addrspace(1)* @__spirv_BuiltInSubDeviceIDINTEL to i32 addrspace(4)*), align 4
  %1 = load i32, i32 addrspace(4)* addrspacecast (i32 addrspace(1)* @__spirv_BuiltInGlobalHWThreadIDINTEL to i32 addrspace(4)*), align 4
  ; CHECK-LLVM: call spir_func i32 @_Z31__spirv_BuiltInSubDeviceIDINTELv() #1
  ; CHECK-LLVM: call spir_func i32 @_Z36__spirv_BuiltInGlobalHWThreadIDINTELv() #1
  ret void
}
