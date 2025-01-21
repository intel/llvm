; This test is to ensure that SPIR-V Backend is able to generate SPIR-V code
; for Translator, and the code can be translated back to LLVM IR. The source
; code contains instructions which require the SPV_KHR_uniform_group_instructions
; extension.

; If LLVM is built without SPIR-V Backend support this test must pass as well.

; RUN: llvm-as %s -o %t.bc

; The following is to test that 
; RUN: llvm-spirv %t.bc -spirv-text -o %t.spv.txt -spirv-ext=+SPV_KHR_uniform_group_instructions --spirv-use-llvm-backend-target
; RUN: FileCheck < %t.spv.txt %s --check-prefix=CHECK-SPIRV

; RUN: llvm-spirv %t.bc -o %t.spv -spirv-ext=+SPV_KHR_uniform_group_instructions --spirv-use-llvm-backend-target
; RUN: llvm-spirv -r %t.spv -o %t.rev.bc
; RUN: llvm-dis %t.rev.bc
; RUN: FileCheck < %t.rev.ll %s --check-prefix=CHECK-LLVM-SPV

; CHECK-SPIRV: Capability GroupUniformArithmeticKHR
; CHECK-SPIRV: Extension "SPV_KHR_uniform_group_instructions"
; CHECK-SPIRV: GroupBitwiseAndKHR

; CHECK-LLVM-SPV: @test1()
; CHECK-LLVM-SPV: @test2()

target triple = "spir64-unknown-unknown"

define dso_local spir_func void @test1() {
entry:
  %res1 = tail call spir_func i32 @_Z26__spirv_GroupBitwiseAndKHR(i32 2, i32 0, i32 0)
  ret void
}

define dso_local spir_func void @test2() {
entry:
  %res1 = tail call spir_func i32  @_Z21work_group_reduce_andi(i32 0)
  ret void
}

declare dso_local spir_func i32  @_Z26__spirv_GroupBitwiseAndKHR(i32, i32, i32)
declare dso_local spir_func i32  @_Z21work_group_reduce_andi(i32)
