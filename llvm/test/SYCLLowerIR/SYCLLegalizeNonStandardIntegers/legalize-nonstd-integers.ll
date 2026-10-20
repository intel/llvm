; Test legalization of non-standard integer types (i6, i24, i48, etc.).
; Non-standard widths should be widened to the next power-of-2 standard width.
;
; RUN: opt -passes=sycl-legalize-nonstandard-integers -S < %s | FileCheck %s

target triple = "spir64-unknown-unknown"

; Test the vec<half,3> -> i48 -> i64 bitcast pattern
define spir_kernel void @test_i48_bitcast(ptr addrspace(1) %out) {
entry:
  %vec = load <3 x half>, ptr addrspace(1) %out, align 8
  %i48 = bitcast <3 x half> %vec to i48
  %i64 = zext i48 %i48 to i64
  store i64 %i64, ptr addrspace(1) %out, align 8
  ret void
}

; Test the trunc-op-zext pattern with various widths
define spir_kernel void @test_trunc_op_zext(ptr addrspace(1) %out, i32 %in) {
entry:
  ; i6 -> should widen to i8
  %v6 = trunc i32 %in to i6
  %v6_op = add i6 %v6, 5
  %r6 = zext i6 %v6_op to i32

  ; i24 -> should widen to i32
  %v24 = trunc i32 %in to i24
  %v24_op = add i24 %v24, 100
  %r24 = zext i24 %v24_op to i32

  %sum = add i32 %r6, %r24
  store i32 %sum, ptr addrspace(1) %out, align 4
  ret void
}

; After legalization, no non-standard integer operations should remain
; CHECK-LABEL: define spir_kernel void @test_i48_bitcast
; CHECK-NOT: bitcast{{.*}}i48
; CHECK-NOT: zext i48
;
; CHECK-LABEL: define spir_kernel void @test_trunc_op_zext
; CHECK-NOT: trunc{{.*}}i6
; CHECK-NOT: trunc{{.*}}i24
; CHECK-NOT: add i6
; CHECK-NOT: add i24
; CHECK-NOT: zext i6
; CHECK-NOT: zext i24
