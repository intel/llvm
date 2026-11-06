; Test atomicrmw uinc_wrap/udec_wrap with AMDGPU metadata roundtrip.
; The writer emits FunctionCall to __translate_spirv_atomic_uinc_wrap_* /
; __translate_spirv_atomic_udec_wrap_* with InstructionMetadata AuxData. On AMD
; targets the reader converts them back to atomicrmw uinc_wrap/udec_wrap, and
; metadata is restored from AuxData.

; RUN: llvm-spirv --spirv-ext=+SPV_KHR_relaxed_extended_instruction %s -spirv-text --spirv-preserve-auxdata --spirv-max-version=1.5 -o - | FileCheck %s --check-prefixes=CHECK-SPIRV,CHECK-SPIRV-EXT

; RUN: llvm-spirv --spirv-ext=+SPV_KHR_relaxed_extended_instruction %s -o %t.spv --spirv-preserve-auxdata --spirv-max-version=1.5
; RUN: llvm-spirv -r --spirv-preserve-auxdata --spirv-target-triple=spirv64-amd-amdhsa %t.spv -o %t.rev.bc
; RUN: llvm-dis %t.rev.bc -o - | FileCheck %s --check-prefix=CHECK-LLVM
; RUN: llvm-spirv -r %t.spv -o %t.rev.without.bc
; RUN: llvm-dis %t.rev.without.bc -o - | FileCheck %s --implicit-check-not="{{amdgpu.no.fine.grained.memory|amdgpu.no.remote.memory}}"

; RUN: llvm-spirv --spirv-ext=+SPV_KHR_relaxed_extended_instruction %s -spirv-text --spirv-preserve-auxdata -o - | FileCheck %s --check-prefixes=CHECK-SPIRV,CHECK-SPIRV-NOEXT

; RUN: llvm-spirv --spirv-ext=+SPV_KHR_relaxed_extended_instruction %s -o %t.spv --spirv-preserve-auxdata
; RUN: llvm-spirv -r --spirv-preserve-auxdata --spirv-target-triple=spirv64-amd-amdhsa %t.spv -o %t.rev.bc
; RUN: llvm-dis %t.rev.bc -o - | FileCheck %s --check-prefix=CHECK-LLVM
; RUN: llvm-spirv -r %t.spv -o %t.rev.without.bc
; RUN: llvm-dis %t.rev.without.bc -o - | FileCheck %s --implicit-check-not="{{amdgpu.no.fine.grained.memory|amdgpu.no.remote.memory}}"

; RUN: llvm-spirv %s -spirv-text -o - | FileCheck %s --check-prefix=CHECK-NO-AUXDATA

; RUN: llvm-spirv %s -o %t.noaux.spv
; RUN: spirv-val %t.noaux.spv

; RUN: llvm-spirv --spirv-ext=+SPV_KHR_relaxed_extended_instruction %s -o %t.aux.spv --spirv-preserve-auxdata
; RUN: spirv-val %t.aux.spv

; RUN: not llvm-spirv %s -spirv-text --spirv-preserve-auxdata --spirv-max-version=1.5 --spirv-ext=+SPV_KHR_relaxed_extended_instruction,-SPV_KHR_non_semantic_info -o - 2>&1 | FileCheck %s --check-prefix=CHECK-EXT-DISABLED

; InstructionMetadata is emitted only if SPV_KHR_relaxed_extended_instruction is allowed.
; RUN: llvm-spirv %s -spirv-text --spirv-preserve-auxdata -o - | FileCheck %s --check-prefix=CHECK-NO-RELAXED
; RUN: llvm-spirv %s -o %t.norelaxed.spv --spirv-preserve-auxdata
; RUN: spirv-val %t.norelaxed.spv

; CHECK-NO-RELAXED-NOT: SPV_KHR_relaxed_extended_instruction
; CHECK-NO-RELAXED-NOT: amdgpu.no.fine.grained.memory
; CHECK-NO-RELAXED-NOT: amdgpu.no.remote.memory
; CHECK-NO-RELAXED-NOT: NonSemanticAuxDataInstructionMetadata

; CHECK-NO-AUXDATA-NOT: NonSemantic.AuxData
; CHECK-NO-AUXDATA-NOT: amdgpu.no.fine.grained.memory
; CHECK-NO-AUXDATA-NOT: amdgpu.no.remote.memory
; CHECK-NO-AUXDATA-NOT: NonSemanticAuxDataInstructionMetadata

; CHECK-EXT-DISABLED: RequiresExtension: Feature requires the following SPIR-V extension:
; CHECK-EXT-DISABLED-NEXT: SPV_KHR_non_semantic_info

; CHECK-SPIRV-EXT: 119734787 65536
; CHECK-SPIRV-EXT: Extension "SPV_KHR_non_semantic_info"
; CHECK-SPIRV-NOEXT: 119734787 67072

; InstructionMetadata forward-references its target, so it is emitted as
; OpExtInstWithForwardRefsKHR, which requires this extension in every version.
; CHECK-SPIRV: Extension "SPV_KHR_relaxed_extended_instruction"

; CHECK-SPIRV-DAG: ExtInstImport [[#Import:]] "NonSemantic.AuxData"
; CHECK-SPIRV-DAG: String [[#MD_NFG:]] "amdgpu.no.fine.grained.memory"
; CHECK-SPIRV-DAG: String [[#MD_NRM:]] "amdgpu.no.remote.memory"
; CHECK-SPIRV-DAG: TypeVoid [[#VoidT:]]
; CHECK-SPIRV-DAG: Decorate [[#UIncFn:]] LinkageAttributes "__translate_spirv_atomic_uinc_wrap_p1_i32" Import
; CHECK-SPIRV-DAG: Decorate [[#UDecFn:]] LinkageAttributes "__translate_spirv_atomic_udec_wrap_p1_i32" Import

; CHECK-SPIRV-DAG: ExtInstWithForwardRefsKHR [[#VoidT]] [[#]] [[#Import]] NonSemanticAuxDataInstructionMetadata [[#UIncRes:]] [[#MD_NFG]]
; CHECK-SPIRV-DAG: ExtInstWithForwardRefsKHR [[#VoidT]] [[#]] [[#Import]] NonSemanticAuxDataInstructionMetadata [[#UIncRes]] [[#MD_NRM]]
; CHECK-SPIRV-DAG: ExtInstWithForwardRefsKHR [[#VoidT]] [[#]] [[#Import]] NonSemanticAuxDataInstructionMetadata [[#UDecRes:]] [[#MD_NFG]]

; CHECK-SPIRV: FunctionCall [[#]] [[#UIncRes]] [[#UIncFn]]
; CHECK-SPIRV: FunctionCall [[#]] [[#UDecRes]] [[#UDecFn]]

target datalayout = "e-i64:64-v16:16-v24:32-v32:32-v48:64-v96:128-v192:256-v256:256-v512:512-v1024:1024"
target triple = "spirv64-amd-amdhsa"

@ui = common dso_local addrspace(1) global i32 0, align 4

; CHECK-LLVM: atomicrmw uinc_wrap ptr addrspace(1) @ui, i32 42 seq_cst
; CHECK-LLVM-SAME: !amdgpu.no.fine.grained.memory ![[#]]
; CHECK-LLVM-SAME: !amdgpu.no.remote.memory ![[#]]
define amdgpu_kernel void @test_uinc_wrap() {
  %uinc = atomicrmw uinc_wrap ptr addrspace(1) @ui, i32 42 seq_cst, !amdgpu.no.fine.grained.memory !0, !amdgpu.no.remote.memory !0
  ret void
}

; CHECK-LLVM: atomicrmw udec_wrap ptr addrspace(1) @ui, i32 42 seq_cst
; CHECK-LLVM-SAME: !amdgpu.no.fine.grained.memory ![[#]]
; CHECK-LLVM-NOT: !amdgpu.no.remote.memory
define amdgpu_kernel void @test_udec_wrap() {
  %udec = atomicrmw udec_wrap ptr addrspace(1) @ui, i32 42 seq_cst, !amdgpu.no.fine.grained.memory !0
  ret void
}

!0 = !{}
