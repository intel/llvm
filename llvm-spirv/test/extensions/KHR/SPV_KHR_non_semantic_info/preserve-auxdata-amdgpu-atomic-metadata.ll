; Test AMDGPU atomic metadata roundtrip via NonSemantic.AuxData InstructionMetadata.

; RUN: llvm-spirv --spirv-ext=+SPV_EXT_shader_atomic_float_add,+SPV_KHR_relaxed_extended_instruction %s -spirv-text --spirv-preserve-auxdata --spirv-max-version=1.5 -o - | FileCheck %s --check-prefixes=CHECK-SPIRV,CHECK-SPIRV-EXT

; RUN: llvm-spirv --spirv-ext=+SPV_EXT_shader_atomic_float_add,+SPV_KHR_relaxed_extended_instruction %s -o %t.spv --spirv-preserve-auxdata --spirv-max-version=1.5
; RUN: llvm-spirv %t.spv -to-text -o - | FileCheck %s --check-prefixes=CHECK-SPIRV,CHECK-SPIRV-EXT
; RUN: llvm-spirv -r --spirv-preserve-auxdata --spirv-target-triple=spirv64-amd-amdhsa %t.spv -o %t.rev.bc
; RUN: llvm-dis %t.rev.bc -o - | FileCheck %s --check-prefix=CHECK-LLVM
; RUN: llvm-spirv -r %t.spv -o %t.rev.without.bc
; RUN: llvm-dis %t.rev.without.bc -o - | FileCheck %s --implicit-check-not="{{amdgpu.no.fine.grained.memory|amdgpu.no.remote.memory|atomic.ignore.denormal.mode}}"

; RUN: llvm-spirv --spirv-ext=+SPV_EXT_shader_atomic_float_add,+SPV_KHR_relaxed_extended_instruction %s -spirv-text --spirv-preserve-auxdata -o - | FileCheck %s --check-prefixes=CHECK-SPIRV,CHECK-SPIRV-NOEXT

; RUN: llvm-spirv --spirv-ext=+SPV_EXT_shader_atomic_float_add,+SPV_KHR_relaxed_extended_instruction %s -o %t.spv --spirv-preserve-auxdata
; RUN: llvm-spirv %t.spv -to-text -o - | FileCheck %s --check-prefixes=CHECK-SPIRV,CHECK-SPIRV-NOEXT
; RUN: llvm-spirv -r --spirv-preserve-auxdata --spirv-target-triple=spirv64-amd-amdhsa %t.spv -o %t.rev.bc
; RUN: llvm-dis %t.rev.bc -o - | FileCheck %s --check-prefix=CHECK-LLVM
; RUN: llvm-spirv -r %t.spv -o %t.rev.without.bc
; RUN: llvm-dis %t.rev.without.bc -o - | FileCheck %s --implicit-check-not="{{amdgpu.no.fine.grained.memory|amdgpu.no.remote.memory|atomic.ignore.denormal.mode}}"

; RUN: llvm-spirv --spirv-ext=+SPV_EXT_shader_atomic_float_add %s -spirv-text -o - | FileCheck %s --check-prefix=CHECK-NO-AUXDATA

; RUN: llvm-spirv --spirv-ext=+SPV_EXT_shader_atomic_float_add %s -o %t.noaux.spv
; RUN: spirv-val %t.noaux.spv

; RUN: llvm-spirv --spirv-ext=+SPV_EXT_shader_atomic_float_add,+SPV_KHR_relaxed_extended_instruction %s -o %t.aux.spv --spirv-preserve-auxdata
; RUN: spirv-val %t.aux.spv

; RUN: not llvm-spirv --spirv-ext=+SPV_EXT_shader_atomic_float_add,+SPV_KHR_relaxed_extended_instruction,-SPV_KHR_non_semantic_info %s -spirv-text --spirv-preserve-auxdata --spirv-max-version=1.5 -o - 2>&1 | FileCheck %s --check-prefix=CHECK-EXT-DISABLED

; InstructionMetadata is emitted only if SPV_KHR_relaxed_extended_instruction is allowed.
; RUN: llvm-spirv --spirv-ext=+SPV_EXT_shader_atomic_float_add %s -spirv-text --spirv-preserve-auxdata -o - | FileCheck %s --check-prefix=CHECK-NO-RELAXED
; RUN: llvm-spirv --spirv-ext=+SPV_EXT_shader_atomic_float_add %s -o %t.norelaxed.spv --spirv-preserve-auxdata
; RUN: spirv-val %t.norelaxed.spv

; CHECK-NO-RELAXED-NOT: SPV_KHR_relaxed_extended_instruction
; CHECK-NO-RELAXED-NOT: amdgpu.no.fine.grained.memory
; CHECK-NO-RELAXED-NOT: amdgpu.no.remote.memory
; CHECK-NO-RELAXED-NOT: atomic.ignore.denormal.mode
; CHECK-NO-RELAXED-NOT: NonSemanticAuxDataInstructionMetadata

; CHECK-NO-AUXDATA-NOT: NonSemantic.AuxData
; CHECK-NO-AUXDATA-NOT: amdgpu.no.fine.grained.memory
; CHECK-NO-AUXDATA-NOT: amdgpu.no.remote.memory
; CHECK-NO-AUXDATA-NOT: atomic.ignore.denormal.mode
; CHECK-NO-AUXDATA-NOT: NonSemanticAuxDataInstructionMetadata

; CHECK-EXT-DISABLED: RequiresExtension: Feature requires the following SPIR-V extension:
; CHECK-EXT-DISABLED-NEXT: SPV_KHR_non_semantic_info

; CHECK-SPIRV-EXT: 119734787 65536
; CHECK-SPIRV-EXT: Extension "SPV_KHR_non_semantic_info"
; CHECK-SPIRV-NOEXT: 119734787 67072

; InstructionMetadata forward-references its target, so it is emitted as
; OpExtInstWithForwardRefsKHR, which requires this extension in every version.
; CHECK-SPIRV: Extension "SPV_KHR_relaxed_extended_instruction"

; CHECK-SPIRV: ExtInstImport [[#Import:]] "NonSemantic.AuxData"

; CHECK-SPIRV-DAG: String [[#MD_NFG:]] "amdgpu.no.fine.grained.memory"
; CHECK-SPIRV-DAG: String [[#MD_NRM:]] "amdgpu.no.remote.memory"
; CHECK-SPIRV-DAG: String [[#MD_IDN:]] "atomic.ignore.denormal.mode"

; CHECK-SPIRV: TypeVoid [[#VoidT:]]

; CHECK-SPIRV-DAG: ExtInstWithForwardRefsKHR [[#VoidT]] [[#]] [[#Import]] NonSemanticAuxDataInstructionMetadata [[#AddRes:]] [[#MD_NFG]]
; CHECK-SPIRV-DAG: ExtInstWithForwardRefsKHR [[#VoidT]] [[#]] [[#Import]] NonSemanticAuxDataInstructionMetadata [[#AddRes]] [[#MD_NRM]]
; CHECK-SPIRV-DAG: ExtInstWithForwardRefsKHR [[#VoidT]] [[#]] [[#Import]] NonSemanticAuxDataInstructionMetadata [[#FAddRes:]] [[#MD_NFG]]
; CHECK-SPIRV-DAG: ExtInstWithForwardRefsKHR [[#VoidT]] [[#]] [[#Import]] NonSemanticAuxDataInstructionMetadata [[#FAddRes]] [[#MD_NRM]]
; CHECK-SPIRV-DAG: ExtInstWithForwardRefsKHR [[#VoidT]] [[#]] [[#Import]] NonSemanticAuxDataInstructionMetadata [[#FAddRes]] [[#MD_IDN]]
; CHECK-SPIRV-DAG: ExtInstWithForwardRefsKHR [[#VoidT]] [[#]] [[#Import]] NonSemanticAuxDataInstructionMetadata [[#XchgRes:]] [[#MD_NFG]]

; CHECK-SPIRV: AtomicIAdd [[#]] [[#AddRes]]
; CHECK-SPIRV: AtomicFAddEXT [[#]] [[#FAddRes]]
; CHECK-SPIRV: AtomicExchange [[#]] [[#XchgRes]]

target datalayout = "e-i64:64-v16:16-v24:32-v32:32-v48:64-v96:128-v192:256-v256:256-v512:512-v1024:1024"
target triple = "spirv64-amd-amdhsa"

; CHECK-LLVM: call spir_func i32 @_Z10atomic_addPU3AS1Vii({{.*}}){{.*}}!amdgpu.no.fine.grained.memory ![[#]]{{.*}}!amdgpu.no.remote.memory ![[#]]
define amdgpu_kernel void @test_iadd(ptr addrspace(1) %ptr) {
  %add = atomicrmw add ptr addrspace(1) %ptr, i32 1 syncscope("agent") monotonic, !amdgpu.no.fine.grained.memory !0, !amdgpu.no.remote.memory !0
  ret void
}

; CHECK-LLVM: call spir_func float @_Z10atomic_addPU3AS1Vff({{.*}}){{.*}}!atomic.ignore.denormal.mode ![[#]]{{.*}}!amdgpu.no.fine.grained.memory ![[#]]{{.*}}!amdgpu.no.remote.memory ![[#]]
define amdgpu_kernel void @test_fadd(ptr addrspace(1) %ptr) {
  %fadd = atomicrmw fadd ptr addrspace(1) %ptr, float 1.0 syncscope("agent") monotonic, !amdgpu.no.fine.grained.memory !0, !amdgpu.no.remote.memory !0, !atomic.ignore.denormal.mode !0
  ret void
}

; CHECK-LLVM: call spir_func i32 @_Z11atomic_xchgPU3AS1Vii({{.*}}){{.*}}!amdgpu.no.fine.grained.memory ![[#]]
; CHECK-LLVM-NOT: !amdgpu.no.remote.memory
; CHECK-LLVM-NOT: !atomic.ignore.denormal.mode
define amdgpu_kernel void @test_xchg(ptr addrspace(1) %ptr) {
  %xchg = atomicrmw xchg ptr addrspace(1) %ptr, i32 1 syncscope("agent") monotonic, !amdgpu.no.fine.grained.memory !0
  ret void
}

!0 = !{}
