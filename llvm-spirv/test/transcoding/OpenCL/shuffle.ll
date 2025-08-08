; RUN: llvm-as %s -o %t.bc
; RUN: llvm-spirv %t.bc -o %t.spv
; RUN: llvm-spirv %t.spv -to-text -o %t.spt
; RUN: FileCheck < %t.spt %s --check-prefix=CHECK-SPIRV

; RUN: llvm-spirv -r %t.spv -o %t.rev.bc
; RUN: llvm-dis < %t.rev.bc | FileCheck %s --check-prefixes=CHECK-LLVM
; RUN: llvm-spirv -r %t.spv --spirv-target-env=SPV-IR -o %t.rev.bc
; RUN: llvm-dis < %t.rev.bc | FileCheck %s --check-prefixes=CHECK-SPV-IR

; Check OpenCL built-in shuffle and shuffle2 translation.

target datalayout = "e-i64:64-v16:16-v24:32-v32:32-v48:64-v96:128-v192:256-v256:256-v512:512-v1024:1024-n8:16:32:64-G1"
target triple = "spir64"

; CHECK-SPIRV: ExtInst [[#]] [[#]] [[#]] shuffle
; CHECK-SPIRV: ExtInst [[#]] [[#]] [[#]] shuffle2

; CHECK-LLVM: call spir_func <2 x float> @_Z7shuffleDv2_fDv2_j(
; CHECK-LLVM: call spir_func <4 x float> @_Z8shuffle2Dv2_fS_Dv4_j(

; CHECK-SPV-IR: call spir_func <2 x float> @_Z19__spirv_ocl_shuffleDv2_fDv2_j(
; CHECK-SPV-IR: call spir_func <4 x float> @_Z20__spirv_ocl_shuffle2Dv2_fS_Dv4_j(

target datalayout = "e-i64:64-v16:16-v24:32-v32:32-v48:64-v96:128-v192:256-v256:256-v512:512-v1024:1024-n8:16:32:64-G1"
target triple = "spir64"

define spir_kernel void @test() {
entry:
  %call = call spir_func <2 x float> @_Z7shuffleDv2_fDv2_j(<2 x float> zeroinitializer, <2 x i32> zeroinitializer)
  ret void
}

declare spir_func <2 x float> @_Z7shuffleDv2_fDv2_j(<2 x float>, <2 x i32>)

define spir_kernel void @test2() {
entry:
  %call = call spir_func <4 x float> @_Z8shuffle2Dv2_fS_Dv4_j(<2 x float> zeroinitializer, <2 x float> zeroinitializer, <4 x i32> zeroinitializer)
  ret void
}

declare spir_func <4 x float> @_Z8shuffle2Dv2_fS_Dv4_j(<2 x float>, <2 x float>, <4 x i32>)

!opencl.ocl.version = !{!0}

!0 = !{i32 3, i32 0}
