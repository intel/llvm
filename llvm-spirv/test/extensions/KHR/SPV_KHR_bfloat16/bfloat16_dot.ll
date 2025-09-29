; RUN: llvm-as %s -o %t.bc
; RUN: llvm-spirv %t.bc --spirv-ext=+SPV_KHR_bfloat16 -o %t.spv
; RUN: llvm-spirv %t.spv -to-text -o - | FileCheck %s --check-prefix=CHECK-SPIRV

; RUN: llvm-spirv -r %t.spv -o %t.rev.bc
; RUN: llvm-dis %t.rev.bc -o - | FileCheck %s --check-prefix=CHECK-LLVM

source_filename = "bfloat16_dot.cpp"
target datalayout = "e-i64:64-v16:16-v24:32-v32:32-v48:64-v96:128-v192:256-v256:256-v512:512-v1024:1024-n8:16:32:64"
target triple = "spirv64-unknown-unknown"

; CHECK-SPIRV-DAG: Capability BFloat16TypeKHR
; CHECK-SPIRV-DAG: Capability BFloat16DotProductKHR
; CHECK-SPIRV-DAG: Extension "SPV_KHR_bfloat16"
; CHECK-SPIRV: 4 TypeFloat [[BFLOAT:[0-9]+]] 16 0
; CHECK-SPIRV: 4 TypeVector [[#]] [[BFLOAT]] 2
; CHECK-SPIRV: Dot

; CHECK-LLVM: %addrA = alloca <2 x bfloat>
; CHECK-LLVM: %addrB = alloca <2 x bfloat>
; CHECK-LLVM: %dataA = load <2 x bfloat>, ptr %addrA
; CHECK-LLVM: %dataB = load <2 x bfloat>, ptr %addrB
; CHECK-LLVM: %call = call spir_func bfloat @_Z3dotDv2_u6__bf16S_(<2 x bfloat> %dataA, <2 x bfloat> %dataB)

declare spir_func bfloat @_Z3dotDv2_u6__bf16Dv2_S_(<2 x bfloat>, <2 x bfloat>)

define spir_kernel void @test() {
entry:
  %addrA = alloca <2 x bfloat>
  %addrB = alloca <2 x bfloat>
  %dataA = load <2 x bfloat>, ptr %addrA
  %dataB = load <2 x bfloat>, ptr %addrB
  %call = call spir_func bfloat @_Z3dotDv2_u6__bf16Dv2_S_(<2 x bfloat> %dataA, <2 x bfloat> %dataB)
  ret void
}

!opencl.ocl.version = !{!7}

!7 = !{i32 2, i32 0}
