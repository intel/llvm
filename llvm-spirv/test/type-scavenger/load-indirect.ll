; RUN: llvm-as %s -o %t.bc
; RUN: llvm-spirv %t.bc -spirv-text -o - | FileCheck %s
; RUN: llvm-spirv %t.bc -o %t.spv
; RUN: spirv-val %t.spv

target datalayout = "e-p:32:32-i64:64-v16:16-v24:32-v32:32-v48:64-v96:128-v192:256-v256:256-v512:512-v1024:1024"
target triple = "spir-unknown-unknown"

; CHECK-DAG: 4 TypeInt [[INT:[0-9]+]] 32 0
; CHECK-DAG: 4 TypeInt [[CHAR:[0-9]+]] 8 0
; CHECK-DAG: 3 TypeFloat [[FLOAT:[0-9]+]] 32
; CHECK-DAG: 4 TypePointer [[INTPTR:[0-9]+]] 7 [[INT]]
; CHECK-DAG: 4 TypePointer [[FLOATPTR:[0-9]+]] 7 [[FLOAT]]
; CHECK-DAG: 4 TypePointer [[CHARPTR:[0-9]+]] 7 [[CHAR]]
; CHECK-DAG: 4 TypePointer [[INTPPTR:[0-9]+]] 7 [[INTPTR]]
; CHECK-DAG: 4 TypePointer [[FLOATPPTR:[0-9]+]] 7 [[FLOATPTR]]
; CHECK-DAG: 4 TypePointer [[CHARPPTR:[0-9]+]] 7 [[CHARPTR]]

; Function Attrs: nounwind
define spir_kernel void @foo() {
; CHECK: 4 Variable [[INTPTR]] [[IPTR:[0-9]+]] 7
; CHECK: 4 Variable [[CHARPPTR]] [[PPTR:[0-9]+]] 7
; CHECK: 4 Bitcast [[INTPPTR]] [[STOREPTR1:[0-9]+]] [[PPTR]]
; CHECK: Store [[STOREPTR1]] [[IPTR]]
; CHECK: 4 Bitcast [[FLOATPPTR]] [[LOADPTR1:[0-9]+]] [[PPTR]]
; CHECK: Load [[FLOATPTR]] [[LOAD1:[0-9]+]] [[LOADPTR1]]
; CHECK: Load [[FLOAT]] [[LOAD2:[0-9]+]] [[LOAD1]]
entry:
  %iptr = alloca i32, align 4
  %pptr = alloca ptr, align 4
  store ptr %iptr, ptr %pptr, align 8
  %0 = load ptr, ptr %pptr, align 8
  %1 = load float, ptr %0, align 4
  store ptr null, ptr %iptr, align 8
  store ptr null, ptr poison, align 8
  store ptr %pptr, ptr %pptr, align 8
  ret void
}
