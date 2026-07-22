; Constant zero-sized memcpy/memmove/memset are no-ops. Emitting
; OpCopyMemorySized with a Size operand of 0 is invalid SPIR-V, so the
; translator must drop these intrinsics. A non-zero copy is unaffected.

; RUN: llvm-spirv %s -spirv-text -o - | FileCheck %s --check-prefix=CHECK-SPIRV
; RUN: llvm-spirv %s -o %t.spv
; RUN: spirv-val %t.spv

target datalayout = "e-p:32:32-i64:64-v16:16-v24:32-v32:32-v48:64-v96:128-v192:256-v256:256-v512:512-v1024:1024-n8:16:32:64"
target triple = "spir-unknown-unknown"

; A single CopyMemorySized is emitted, for the non-zero memcpy only.
; CHECK-SPIRV-COUNT-1: CopyMemorySized

define spir_func void @zero_memcpy(ptr %dst, ptr %src) {
entry:
  call void @llvm.memcpy.p0.p0.i32(ptr align 4 %dst, ptr align 4 %src, i32 0, i1 false)
  ret void
}

define spir_func void @zero_memmove(ptr %dst, ptr %src) {
entry:
  call void @llvm.memmove.p0.p0.i32(ptr align 4 %dst, ptr align 4 %src, i32 0, i1 false)
  ret void
}

define spir_func void @zero_memset(ptr %dst) {
entry:
  call void @llvm.memset.p0.i32(ptr align 4 %dst, i8 7, i32 0, i1 false)
  ret void
}

define spir_func void @nonzero_memcpy(ptr %dst, ptr %src) {
entry:
  call void @llvm.memcpy.p0.p0.i32(ptr align 4 %dst, ptr align 4 %src, i32 16, i1 false)
  ret void
}

declare void @llvm.memcpy.p0.p0.i32(ptr captures(none), ptr captures(none) readonly, i32, i1)
declare void @llvm.memmove.p0.p0.i32(ptr captures(none), ptr captures(none) readonly, i32, i1)
declare void @llvm.memset.p0.i32(ptr captures(none), i8, i32, i1)
