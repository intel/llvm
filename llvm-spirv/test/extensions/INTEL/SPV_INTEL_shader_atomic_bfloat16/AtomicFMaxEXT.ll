; RUN: llvm-spirv %s --spirv-ext=+SPV_INTEL_shader_atomic_bfloat16,+SPV_KHR_bfloat16 -o %t.spv
; RUN: llvm-spirv -to-text %t.spv -o %t.spt
; RUN: FileCheck < %t.spt %s --check-prefix=CHECK-SPIRV

; RUN: llvm-spirv --spirv-target-env=SPV-IR -r %t.spv -o %t.rev.bc
; RUN: llvm-dis %t.rev.bc -o - | FileCheck %s --check-prefixes=CHECK-LLVM-SPV

target datalayout = "e-i64:64-v16:16-v24:32-v32:32-v48:64-v96:128-v192:256-v256:256-v512:512-v1024:1024-n8:16:32:64"
target triple = "spir64-unknown-unknown"

; CHECK-SPIRV-DAG: Capability AtomicBFloat16MinMaxINTEL
; CHECK-SPIRV-DAG: Capability BFloat16TypeKHR
; CHECK-SPIRV-DAG: Extension "SPV_INTEL_shader_atomic_bfloat16"
; CHECK-SPIRV-DAG: Extension "SPV_KHR_bfloat16"

; CHECK-SPIRV: TypeFloat [[BFLOAT:[0-9]+]] 16 0

; Function Attrs: convergent norecurse nounwind
define dso_local spir_func bfloat @test_AtomicFMaxEXT_bfloat(ptr addrspace(4) align 2 dereferenceable(4) %Arg) {
entry:
  %0 = addrspacecast ptr addrspace(4) %Arg to ptr addrspace(1)
  ; CHECK-SPIRV: AtomicFMaxEXT [[BFLOAT]]
  ; CHECK-LLVM-SPV: call spir_func bfloat @_Z21__spirv_AtomicFMaxEXTPU3AS1DF16biiDF16b({{.*}}bfloat
  %ret = tail call spir_func bfloat @_Z21__spirv_AtomicFMaxEXTPU3AS1DF16biiDF16b(ptr addrspace(1) %0, i32 1, i32 896, bfloat 1.000000e+00)
  ret bfloat %ret
}

; Function Attrs: convergent
declare dso_local spir_func bfloat @_Z21__spirv_AtomicFMaxEXTPU3AS1DF16biiDF16b(ptr addrspace(1), i32, i32, bfloat)
