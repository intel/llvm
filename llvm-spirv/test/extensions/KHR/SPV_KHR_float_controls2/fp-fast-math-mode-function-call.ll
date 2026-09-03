; Verify that FPFastMathMode decorations are emitted for OpFunctionCall when
; SPV_KHR_float_controls2 is enabled. Also verify that non-core instructions
; (e.g. OpGroupFMulKHR from SPV_KHR_uniform_group_instructions) do NOT get a
; spurious FPFastMathMode decoration. SPV_KHR_float_controls2 extends
; FPFastMathMode to all *core* instructions, but not to extension-defined ones.

; Without FC2: no decoration on call.
; RUN: llvm-spirv -spirv-text %s --spirv-max-version=1.5 --spirv-ext=+SPV_KHR_uniform_group_instructions -o - | FileCheck %s --check-prefix=NO-DECO
; RUN: llvm-spirv -spirv-text %s --spirv-ext=+SPV_KHR_uniform_group_instructions -o - | FileCheck %s --check-prefix=NO-DECO

; With FC2: decoration emitted on OpFunctionCall, not on extension instructions.
; RUN: llvm-spirv -spirv-text %s --spirv-ext=+SPV_KHR_float_controls2,+SPV_KHR_uniform_group_instructions -o - | FileCheck %s --check-prefix=SPIRV

; Roundtrip with FC2.
; RUN: llvm-spirv %s --spirv-ext=+SPV_KHR_float_controls2,+SPV_KHR_uniform_group_instructions -o %t.spv
; RUN: spirv-val %t.spv
; RUN: llvm-spirv -r %t.spv -o - | llvm-dis | FileCheck %s --check-prefix=IR

target triple = "spirv-unknown-unknown"

define internal spir_func float @helper(float %x) {
  %r = fmul float %x, %x
  ret float %r
}

declare spir_func float @_Z20__spirv_GroupFMulKHRjjf(i32, i32, float)

; NO-DECO-NOT: FPFastMathMode

; SPIRV-DAG: FunctionCall [[#]] [[#CALL:]]
; SPIRV-DAG: GroupFMulKHR [[#]] [[#GFMUL:]]
; SPIRV-DAG: Decorate [[#CALL]] FPFastMathMode 458767

; OpGroupFMulKHR is defined by SPV_KHR_uniform_group_instructions, not core.
; It should NOT get FPFastMathMode because FC2 only covers core instructions.
; SPIRV-NOT: Decorate [[#GFMUL]] FPFastMathMode

define spir_kernel void @test(ptr addrspace(1) %out, float %a) {
entry:
  ; IR: %call_fast = call reassoc nnan ninf nsz arcp contract spir_func float @helper(float %a)
  %call_fast = call fast spir_func float @helper(float %a)
  store float %call_fast, ptr addrspace(1) %out, align 4

  %gfmul_fast = call fast spir_func float @_Z20__spirv_GroupFMulKHRjjf(i32 2, i32 0, float %a)
  store float %gfmul_fast, ptr addrspace(1) %out, align 4

  ret void
}
