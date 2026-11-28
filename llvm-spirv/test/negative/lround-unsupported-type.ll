; RUN: not llvm-spirv %s -o %t.spv 2>&1 | FileCheck %s

; CHECK: InvalidFunctionCall
; CHECK: llvm.lround.i64.f128

target triple = "spir64-unknown-unknown"

define spir_func i64 @f() {
  %r = call i64 @llvm.lround.i64.f128(fp128 0xL00000000000000003FFF000000000000)
  ret i64 %r
}
