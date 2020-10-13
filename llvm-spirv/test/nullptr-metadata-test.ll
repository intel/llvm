; This test ensures that the translator does not crash
; RUN: llvm-as %s -o %t.bc
; RUN: llvm-spirv %t.bc -o %t.spv

; ModuleID = 'test.bc'
target triple = "spir64"

declare dllexport void @test_func(i32) #0

attributes #0 = { "VCSLMSize"="0" }
