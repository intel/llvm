; A VectorCompute module (detected via VCGlobalVariable metadata) keeps using
; VectorAnyINTEL and SPV_INTEL_vector_compute for a non-standard vector size,
; even when SPV_EXT_long_vector is also enabled.

; RUN: llvm-spirv %s --spirv-ext=+SPV_EXT_long_vector,+SPV_INTEL_vector_compute -o %t.spv
; RUN: llvm-spirv %t.spv -to-text -o - | FileCheck %s --implicit-check-not="Capability LongVectorEXT" --implicit-check-not='Extension "SPV_EXT_long_vector"'

; CHECK-DAG: Capability VectorAnyINTEL
; CHECK-DAG: Extension "SPV_INTEL_vector_compute"
; CHECK-DAG: TypeFloat [[#F32:]] 32
; CHECK-DAG: TypeVector [[#]] [[#F32]] 5

target datalayout = "e-i64:64-v16:16-v24:32-v32:32-v48:64-v96:128-v192:256-v256:256-v512:512-v1024:1024"
target triple = "spir64-unknown-unknown"

@gv = internal addrspace(1) global <5 x float> zeroinitializer #0

attributes #0 = { "VCGlobalVariable" }
