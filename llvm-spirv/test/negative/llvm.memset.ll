; RUN: llvm-as %s -o %t.bc
; RUN: not llvm-spirv %t.bc -o %t.spv 2>&1 | FileCheck %s

; CHECK: InvalidFunctionCall: Unexpected llvm intrinsic:
; CHECK: Translation of llvm.memset requires a const `length` argument

target triple = "spir"

define void @spir_func(i8* %ptr, i32 %length) {
  call void @llvm.memset.p0i8.i32(i8* %ptr, i8 42, i32 %length, i1 false)
  ret void
}

declare void @llvm.memset.p0i8.i32(i8*, i8, i32, i1)
