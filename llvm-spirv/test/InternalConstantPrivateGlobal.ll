; RUN: llvm-as %s -o %t.bc
; RUN: llvm-spirv -s %t.bc -o %t.out.bc
; RUN: llvm-dis < %t.out.bc | FileCheck %s

target datalayout = "e-i64:64-v16:16-v24:32-v32:32-v48:64-v96:128-v192:256-v256:256-v512:512-v1024:1024-n8:16:32:64"
target triple = "spir64-unknown-unknown"

; check that all internal private globals are removed
; CHECK-NOT: @g_var_1
; CHECK-NOT: @g_var_2
@g_var_1 = internal unnamed_addr constant [8 x i32] [i32 0, i32 1, i32 -2, i32 3, i32 4, i32 5, i32 6, i32 7]
@g_var_2 = internal unnamed_addr constant [8 x i32] [i32 2, i32 1, i32 -2, i32 3, i32 4, i32 5, i32 6, i32 7]

; check that external global is not changed
; CHECK: @g_var_e
@g_var_e = external unnamed_addr constant [8 x i32]

; check that non constant global is not changed
; CHECK: @g_var_nc = internal global i32 4
@g_var_nc = internal global i32 4

; CHECK-LABEL: define spir_func i32 @foo(
define spir_func i32 @foo(i32 %i) {
  ; CHECK: [[alloca:%.*]] = alloca [8 x i32], align 4
  ; CHECK: store [8 x i32] [i32 0, i32 1, i32 -2, i32 3, i32 4, i32 5, i32 6, i32 7], ptr [[alloca]], align 4
  ; CHECK: [[p:%.*]] = getelementptr [8 x i32], ptr [[alloca]], i32 0, i32 %i
  %p_1 = getelementptr [8 x i32], ptr @g_var_1, i32 0, i32 %i
  %v = load i32, ptr %p_1, align 4

  ; CHECK: %p2 = getelementptr [8 x i32], ptr [[alloca]], i32 0, i32 %i
  %p2 = getelementptr [8 x i32], ptr @g_var_1, i32 0, i32 %i
  %v2 = load i32, ptr %p2, align 4
  %v3 = add i32 %v, %v2

  ret i32 %v3
}

; CHECK-LABEL: define spir_func i32 @bar
define spir_func i32 @bar() {
  ; CHECK: [[alloca:%.*]] = alloca [8 x i32], align 4
  ; CHECK: store [8 x i32] [i32 2, i32 1, i32 -2, i32 3, i32 4, i32 5, i32 6, i32 7], ptr [[alloca]], align 4
  ; CHECK: [[p:%.*]] = getelementptr [8 x i32], ptr [[alloca]], i32 0, i32 3
  %p_1 = getelementptr [8 x i32], ptr @g_var_2, i32 0, i32 3
  %v = load i32, ptr %p_1, align 4
  ret i32 %v
}

; CHECK-LABEL: define spir_func i32 @foobar
define spir_func i32 @foobar() {
  ; CHECK: [[alloca_2:%.*]] = alloca [8 x i32], align 4
  ; CHECK: store [8 x i32] [i32 2, i32 1, i32 -2, i32 3, i32 4, i32 5, i32 6, i32 7], ptr [[alloca_2]], align 4
  ; CHECK: [[alloca_1:%.*]] = alloca [8 x i32], align 4
  ; CHECK: store [8 x i32] [i32 0, i32 1, i32 -2, i32 3, i32 4, i32 5, i32 6, i32 7], ptr [[alloca_1]], align 4
  ; CHECK: [[p_1:%.*]] = getelementptr [8 x i32], ptr [[alloca_1]], i32 0, i32 3
  %p_1 = getelementptr [8 x i32], ptr @g_var_1, i32 0, i32 3
  %v_1 = load i32, ptr %p_1, align 4

  ; CHECK: [[p_2:%.*]] = getelementptr [8 x i32], ptr [[alloca_2]], i32 0, i32 4
  %p_2 = getelementptr [8 x i32], ptr @g_var_2, i32 0, i32 4
  %v_2 = load i32, ptr %p_1, align 4
  %sum = add i32 %v_1, %v_2

  ret i32 %sum
}
