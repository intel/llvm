; RUN: llvm-spirv %s -o %t.spv --spirv-ext=+SPV_INTEL_16bit_atomics
; RUN: llvm-spirv -to-text %t.spv -o %t.spt
; RUN: FileCheck < %t.spt %s --check-prefix=CHECK-SPIRV

; RUN: llvm-spirv -r --spirv-target-env=CL2.0 %t.spv -o %t.rev.bc
; RUN: llvm-dis %t.rev.bc 
; RUN: FileCheck < %t.rev.ll %s --check-prefixes=CHECK-LLVM

; RUN: llvm-spirv -r --spirv-target-env="SPV-IR" %t.spv -o %t.rev.bc
; RUN: llvm-dis %t.rev.bc 
; RUN: FileCheck < %t.rev.ll %s --check-prefixes=CHECK-LLVM-SPV-IR

; Check that without extension we don't use its capabilities - there is no
; limitation on using i16 with atomic instruction in the core specification.
; RUN: llvm-spirv %s -o %t.noext.spv
; RUN: spirv-val %t.noext.spv
; RUN: llvm-spirv -to-text %t.noext.spv -o %t.noext.spt
; RUN: FileCheck < %t.noext.spt %s --check-prefix=CHECK-SPIRV-NOEXT

; CHECK-SPIRV: Capability Int16
; CHECK-SPIRV: Capability AtomicInt16CompareExchangeINTEL
; CHECK-SPIRV: Capability Int16AtomicsINTEL
; CHECK-SPIRV: Extension "SPV_INTEL_16bit_atomics"
; CHECK-SPIRV: AtomicOr

; CHECK-SPIRV-NOEXT: Capability Int16
; CHECK-SPIRV-NOEXT-NOT: Capability AtomicInt16CompareExchangeINTEL
; CHECK-SPIRV-NOEXT-NOT: Capability Int16AtomicsINTEL
; CHECK-SPIRV-NOEXT-NOT: Extension "SPV_INTEL_16bit_atomics"

; CHECK-LLVM: call spir_func i16 @_Z24atomic_fetch_or_explicitPU3AS4VU7_Atomicss12memory_order12memory_scope
; CHECK-LLVM-SPV-IR: call spir_func i16 @_Z16__spirv_AtomicOrPU3AS1siis

target datalayout = "e-i64:64-v16:16-v24:32-v32:32-v48:64-v96:128-v192:256-v256:256-v512:512-v1024:1024"
target triple = "spir64"

@ui = common dso_local addrspace(1) global i16 0, align 4

define dso_local spir_func void @test() {
entry:
  %0 = atomicrmw or ptr addrspace(1) @ui, i16 42 release
  ret void
}
