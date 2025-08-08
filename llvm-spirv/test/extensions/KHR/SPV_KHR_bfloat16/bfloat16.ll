; RUN: llvm-as %s -o %t.bc
; RUN: llvm-spirv %t.bc --spirv-ext=+SPV_KHR_bfloat16 -o %t.spv
; RUN: llvm-spirv %t.spv -to-text -o - | FileCheck %s --check-prefix=CHECK-SPIRV

; RUN: llvm-spirv -r %t.spv -o %t.rev.bc
; RUN: llvm-dis %t.rev.bc -o - | FileCheck %s --check-prefix=CHECK-LLVM

; RUN: not llvm-spirv %t.bc 2>&1 | FileCheck %s --check-prefix=CHECK-ERROR

; CHECK-ERROR: RequiresExtension: Feature requires the following SPIR-V extension:
; CHECK-ERROR-NEXT: SPV_KHR_bfloat16
; CHECK-ERROR-NEXT: NOTE: LLVM module contains bfloat type, translation of which
; CHECK-ERROR-SAME: requires this extension

source_filename = "bfloat16.cpp"
target datalayout = "e-i64:64-v16:16-v24:32-v32:32-v48:64-v96:128-v192:256-v256:256-v512:512-v1024:1024-n8:16:32:64"
target triple = "spirv64-unknown-unknown"

; CHECK-SPIRV-DAG: Capability BFloat16TypeKHR
; CHECK-SPIRV-DAG: Extension "SPV_KHR_bfloat16"
; CHECK-SPIRV: 4 TypeFloat [[BFLOAT:[0-9]+]] 16 0
; CHECK-SPIRV: 4 TypeVector [[#]] [[BFLOAT]] 2

; CHECK-LLVM: [[ADDR1:]] = alloca bfloat
; CHECK-LLVM: [[ADDR2:]] = alloca <2 x bfloat>
; CHECK-LLVM: [[DATA1:]] = load bfloat, ptr [[ADDR1]]
; CHECK-LLVM: [[DATA2:]] = load <2 x bfloat>, ptr [[ADDR2]]

define spir_kernel void @test() {
entry:
  %addr1 = alloca bfloat
  %addr2 = alloca <2 x bfloat>
  %data1 = load bfloat, ptr %addr1
  %data2 = load <2 x bfloat>, ptr %addr2
  ret void
}
