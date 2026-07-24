; RUN: llvm-spirv %s --spirv-ext=+SPV_EXT_long_vector -o %t.spv
; RUN: spirv-val %t.spv
; RUN: llvm-spirv %t.spv -to-text -o - | FileCheck %s

; RUN: llvm-as %s -o %t.bc
; RUN: not llvm-spirv %t.bc 2>&1 | FileCheck %s --check-prefix=ERROR

; CHECK-DAG: Capability LongVectorEXT
; CHECK-DAG: Extension "SPV_EXT_long_vector"
; CHECK-DAG: TypeFloat [[#F32:]] 32
; CHECK-DAG: TypeVector [[#]] [[#F32]] 5

target datalayout = "e-i64:64-v16:16-v24:32-v32:32-v48:64-v96:128-v192:256-v256:256-v512:512-v1024:1024"
target triple = "spir64-unknown-unknown"

define spir_func void @test_vec5(<5 x float> %v) {
entry:
  ret void
}

; ERROR: RequiresExtension: Feature requires the following SPIR-V extension:
; ERROR-NEXT: SPV_EXT_long_vector or SPV_INTEL_vector_compute

define spir_func void @test_no_ext(<5 x float> %v) {
entry:
  ret void
}
