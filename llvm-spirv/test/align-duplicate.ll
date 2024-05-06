; RUN: llvm-as %s -o %t.bc
; RUN: llvm-spirv %t.bc -o %t.spv
; RUN: spirv-val %t.spv

; Test that duplicate align information does not result in SPIR-V validation
; errors due to duplicate Alignment Decorations.

target datalayout = "e-i64:64-v16:16-v24:32-v32:32-v48:64-v96:128-v192:256-v256:256-v512:512-v1024:1024-n8:16:32:64"
target triple = "spir64-unknown-unknown"

define spir_func void @f() {
 %res = alloca i16, align 2, !spirv.Decorations !1
 ret void
}

!1 = !{!2}
!2 = !{i32 44, i32 2}
