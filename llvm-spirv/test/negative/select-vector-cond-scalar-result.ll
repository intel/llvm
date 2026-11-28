; RUN: llvm-as < %s -o %t.bc
; RUN: not llvm-spirv %t.bc -o %t.spv 2>&1 | FileCheck %s

; CHECK: InvalidInstruction: Can't translate llvm instruction:
; CHECK: OpSelect with a vector condition requires a vector result

target datalayout = "e-i64:64-v16:16-v24:32-v32:32-v48:64-v96:128-v192:256-v256:256-v512:512-v1024:1024-n8:16:32:64"
target triple = "spir64-unknown-unknown"

define spir_func i32 @test(<4 x i1> %cond, i32 %a, i32 %b) {
entry:
  %call = call spir_func i32 @_Z14__spirv_SelectDv4_bii(<4 x i1> %cond, i32 %a, i32 %b)
  ret i32 %call
}

declare spir_func i32 @_Z14__spirv_SelectDv4_bii(<4 x i1>, i32, i32)
