; RUN: llvm-as %s -o %t.bc
; RUN: llvm-spirv %t.bc --spirv-ext=+SPV_NV_shader_atomic_fp16_vector -o %t.spv
; RUN: llvm-spirv %t.spv -to-text -o - | FileCheck %s --check-prefix=CHECK-SPIRV
; RUN: llvm-spirv -r %t.spv -o %t.rev.bc
; RUN: llvm-dis %t.rev.bc -o - | FileCheck %s --check-prefix=CHECK-LLVM

; SPV_NV_shader_atomic_fp16_vector lets OpAtomicFAddEXT, OpAtomicFMinEXT and
; OpAtomicFMaxEXT operate on a 2- or 4-component vector of fp16. fsub has no
; atomic opcode, so it is emitted as OpFNegate followed by OpAtomicFAddEXT.

; CHECK-SPIRV-DAG: Capability AtomicFloat16VectorNV
; CHECK-SPIRV-DAG: Extension "SPV_NV_shader_atomic_fp16_vector"
; CHECK-SPIRV: TypeFloat [[#Half:]] 16
; CHECK-SPIRV: TypeVector [[#V2:]] [[#Half]] 2
; CHECK-SPIRV: TypeVector [[#V4:]] [[#Half]] 4
; CHECK-SPIRV: AtomicFAddEXT [[#V2]]
; CHECK-SPIRV: FNegate [[#V2]] [[#Neg:]]
; CHECK-SPIRV: AtomicFAddEXT [[#V2]] [[#]] [[#]] [[#]] [[#]] [[#Neg]]
; CHECK-SPIRV: AtomicFMinEXT [[#V2]]
; CHECK-SPIRV: AtomicFMaxEXT [[#V2]]
; CHECK-SPIRV: AtomicFAddEXT [[#V4]]

; CHECK-LLVM: call spir_func <2 x half> @{{.*}}atomic_add{{.*}}(ptr addrspace(1) %p, <2 x half> %v2)
; CHECK-LLVM: fneg <2 x half> %v2
; CHECK-LLVM: call spir_func <2 x half> @{{.*}}atomic_add
; CHECK-LLVM: call spir_func <2 x half> @{{.*}}atomic_min
; CHECK-LLVM: call spir_func <2 x half> @{{.*}}atomic_max
; CHECK-LLVM: call spir_func <4 x half> @{{.*}}atomic_add

target datalayout = "e-i64:64-v16:16-v24:32-v32:32-v48:64-v96:128-v192:256-v256:256-v512:512-v1024:1024"
target triple = "spir64"

define spir_func void @test(ptr addrspace(1) %p, <2 x half> %v2, <4 x half> %v4) {
entry:
  %a = atomicrmw fadd ptr addrspace(1) %p, <2 x half> %v2 seq_cst
  %s = atomicrmw fsub ptr addrspace(1) %p, <2 x half> %v2 seq_cst
  %mn = atomicrmw fmin ptr addrspace(1) %p, <2 x half> %v2 seq_cst
  %mx = atomicrmw fmax ptr addrspace(1) %p, <2 x half> %v2 seq_cst
  %a4 = atomicrmw fadd ptr addrspace(1) %p, <4 x half> %v4 seq_cst
  ret void
}
