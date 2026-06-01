; Test legalization of non-standard integer types (i6, i24, i48, etc.)
; when the SPV_INTEL_arbitrary_precision_integers extension is disabled.
; Non-standard widths should be widened to the next power-of-2 standard width.
;
; RUN: llvm-as %s -o %t.bc
; RUN: llvm-spirv %t.bc -o %t.spv -spirv-ext=-all,+SPV_KHR_linkonce_odr
; RUN: llvm-spirv -r %t.spv -o %t.rev.bc
; RUN: llvm-dis %t.rev.bc -o - | FileCheck %s --check-prefix=CHECK-LLVM

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

; After round-trip, no non-standard integer operations should remain
; CHECK-LLVM: define spir_kernel void @test_i48_bitcast
; CHECK-LLVM-NOT: bitcast{{.*}}i48
; CHECK-LLVM-NOT: zext i48
;
; CHECK-LLVM: define spir_kernel void @test_trunc_op_zext
; CHECK-LLVM-NOT: trunc{{.*}}i6
; CHECK-LLVM-NOT: trunc{{.*}}i24
; CHECK-LLVM-NOT: add i6
; CHECK-LLVM-NOT: add i24
; CHECK-LLVM-NOT: zext i6
; CHECK-LLVM-NOT: zext i24
