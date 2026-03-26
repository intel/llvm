; RUN: llvm-as < %s -o %t.bc
; RUN: llvm-spirv %t.bc -o %t.spv
; RUN: spirv-val %t.spv
; RUN: llvm-spirv -to-text %t.spv -o - | FileCheck %s
; RUN: llvm-spirv -r %t.spv -o %t.rev.bc
; RUN: llvm-dis < %t.rev.bc | FileCheck %s --check-prefix=CHECK-LLVM
; RUN: %if spirv-backend %{ llc -O0 -mtriple=spirv64-unknown-unknown -filetype=obj %s -o %t.llc.spv %}
; RUN: %if spirv-backend %{ llvm-spirv -r %t.llc.spv -o %t.llc.rev.bc %}
; RUN: %if spirv-backend %{ llvm-dis %t.llc.rev.bc -o %t.llc.rev.ll %}
; RUN: %if spirv-backend %{ FileCheck %s --check-prefix=CHECK-LLVM < %t.llc.rev.ll %}

target datalayout = "e-i64:64-v16:16-v24:32-v32:32-v48:64-v96:128-v192:256-v256:256-v512:512-v1024:1024-n8:16:32:64"
target triple = "spir64-unknown-unknown"

; CHECK: TypeInt [[#Int64Ty:]] 64 0
; CHECK: TypeInt [[#Int32Ty:]] 32 0
; CHECK: Constant [[#Int32Ty]] [[#One32:]] 1
; CHECK: Constant [[#Int64Ty]] [[#ShiftAmt:]] 32 0
; CHECK: TypeVector [[#Vec2Int32Ty:]] [[#Int32Ty]] 2
; CHECK: ConstantComposite [[#Vec2Int32Ty]] [[#VecConst:]] [[#One32]] [[#One32]]
; CHECK: Bitcast [[#Int64Ty]] [[#BitcastedVal:]] [[#VecConst]]
; CHECK: ShiftRightLogical [[#Int64Ty]] [[#Shifted:]] [[#BitcastedVal]] [[#ShiftAmt]]

; CHECK-LLVM: define spir_func void @foo(ptr [[arg:%.*]])
; CHECK-LLVM:   [[bc:%.*]] = bitcast <2 x i32> splat (i32 1) to i64
; CHECK-LLVM:   [[shr:%.*]] = lshr i64 [[bc]], 32
; CHECK-LLVM:   store i64 [[shr]], ptr [[arg]], align 8

define void @foo(ptr %arg) {
entry:
  %0 = lshr i64 bitcast (<2 x i32> <i32 1, i32 1> to i64), 32
  store i64 %0, i64* %arg
  ret void
}
