; RUN: llvm-as %s -o %t.bc
; RUN: llvm-spirv %t.bc --spirv-ext=+SPV_INTEL_kernel_attributes -o %t.spv
; RUN: llvm-spirv %t.spv -to-text -o %t.spt
; RUN: FileCheck < %t.spt %s --check-prefix=CHECK-SPIRV

; RUN: llvm-spirv -r %t.spv -o %t.rev.bc
; RUN: llvm-dis < %t.rev.bc | FileCheck %s --check-prefix=CHECK-LLVM

; FPGAKernelAttributesv2INTEL implicitly defines FPGAKernelAttributesINTEL
; CHECK-SPIRV: Capability FPGAKernelAttributesINTEL
; CHECK-SPIRV: Capability FPGAKernelAttributesv2INTEL
; CHECK-SPIRV: Extension "SPV_INTEL_kernel_attributes"
; CHECK-SPIRV: EntryPoint [[#]] [[#KERNEL1:]] "test_1"
; CHECK-SPIRV: EntryPoint [[#]] [[#KERNEL2:]] "test_2"
; CHECK-SPIRV: ExecutionMode [[#KERNEL1]] 6160 0
; CHECK-SPIRV: ExecutionMode [[#KERNEL2]] 6160 1
; CHECK-SPIRV: Function [[#]] [[#KERNEL1]]
; CHECK-SPIRV: Function [[#]] [[#KERNEL2]]

; CHECK-LLVM: define spir_kernel void @test_1{{.*}} !ip_interface ![[#NOWAITFORDONEWRITE:]]
; CHECK-LLVM: define spir_kernel void @test_2{{.*}} !ip_interface ![[#WAITFORDONEWRITE:]]
; CHECK-LLVM: ![[#NOWAITFORDONEWRITE:]] = !{!"csr"}
; CHECK-LLVM: ![[#WAITFORDONEWRITE:]] = !{!"csr", !"wait_for_done_write"}

; ModuleID = 'test.bc'
target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v16:16:16-v24:32:32-v32:32:32-v48:64:64-v64:64:64-v96:128:128-v128:128:128-v192:256:256-v256:256:256-v512:512:512-v1024:1024:1024"
target triple = "spir64-unknown-unknown"

; Function Attrs: nounwind
define spir_kernel void @test_1() #0 !ip_interface !0
{
entry:
  ret void
}

; Function Attrs: nounwind
define spir_kernel void @test_2() #0 !ip_interface !1
{
entry:
  ret void
}

attributes #0 = { nounwind }

!0 = !{!"csr"}
!1 = !{!"csr", !"wait_for_done_write"}
