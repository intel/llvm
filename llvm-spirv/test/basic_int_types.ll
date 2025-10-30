; RUN: llvm-as %s -o %t.bc
; RUN: llvm-spirv %t.bc -o %t.spv
; RUN: llvm-spirv %t.bc -spirv-text -o %t.txt
; RUN: FileCheck < %t.txt %s
; RUN: spirv-val %t.spv
; RUN: llvm-spirv -r  %t.spv -o %t.rev.bc
; RUN: llvm-dis < %t.rev.bc | FileCheck %s --check-prefix=CHECK-LLVM

target datalayout = "e-i64:64-v16:16-v24:32-v32:32-v48:64-v96:128-v192:256-v256:256-v512:512-v1024:1024"
target triple = "spir64"

define void @main() {
entry:
; CHECK-DAG: TypeInt [[#short:]] 16 0
; CHECK-DAG: TypeInt [[#int:]] 32 0
; CHECK-DAG: TypeInt [[#long:]] 64 0

; CHECK-DAG: TypeVector [[#v2short:]] [[#short]] 2
; CHECK-DAG: TypeVector [[#v3short:]] [[#short]] 3
; CHECK-DAG: TypeVector [[#v4short:]] [[#short]] 4
; CHECK-DAG: TypeVector [[#v2int:]] [[#int]] 2
; CHECK-DAG: TypeVector [[#v3int:]] [[#int]] 3
; CHECK-DAG: TypeVector [[#v4int:]] [[#int]] 4
; CHECK-DAG: TypeVector [[#v2long:]] [[#long]] 2
; CHECK-DAG: TypeVector [[#v3long:]] [[#long]] 3
; CHECK-DAG: TypeVector [[#v4long:]] [[#long]] 4

; CHECK-DAG: TypePointer [[#ptr_Function_short:]] 7 [[#short]]
; CHECK-DAG: TypePointer [[#ptr_Function_int:]] 7 [[#int]]
; CHECK-DAG: TypePointer [[#ptr_Function_long:]] 7 [[#long]]
; CHECK-DAG: TypePointer [[#ptr_Function_v2short:]] 7 [[#v2short]]
; CHECK-DAG: TypePointer [[#ptr_Function_v3short:]] 7 [[#v3short]]
; CHECK-DAG: TypePointer [[#ptr_Function_v4short:]] 7 [[#v4short]]
; CHECK-DAG: TypePointer [[#ptr_Function_v2int:]] 7 [[#v2int]]
; CHECK-DAG: TypePointer [[#ptr_Function_v3int:]] 7 [[#v3int]]
; CHECK-DAG: TypePointer [[#ptr_Function_v4int:]] 7 [[#v4int]]
; CHECK-DAG: TypePointer [[#ptr_Function_v2long:]] 7 [[#v2long]]
; CHECK-DAG: TypePointer [[#ptr_Function_v3long:]] 7 [[#v3long]]
; CHECK-DAG: TypePointer [[#ptr_Function_v4long:]] 7 [[#v4long]]

; CHECK-DAG: Variable [[#ptr_Function_short]] [[#]] 7
; CHECK-LLVM: alloca i16, align 2
  %int16_t_Val = alloca i16, align 2

; CHECK-DAG: Variable [[#ptr_Function_int]] [[#]] 7
; CHECK-LLVM: alloca i32, align 4
  %int_Val = alloca i32, align 4

; CHECK-DAG: Variable [[#ptr_Function_long]] [[#]] 7
; CHECK-LLVM: alloca i64, align 8
  %int64_t_Val = alloca i64, align 8

; CHECK-DAG: Variable [[#ptr_Function_v2short]] [[#]] 7
; CHECK-LLVM: alloca <2 x i16>, align 4
  %int16_t2_Val = alloca <2 x i16>, align 4

; CHECK-DAG: Variable [[#ptr_Function_v3short]] [[#]] 7
; CHECK-LLVM: alloca <3 x i16>, align 8
  %int16_t3_Val = alloca <3 x i16>, align 8

; CHECK-DAG: Variable [[#ptr_Function_v4short]] [[#]] 7
; CHECK-LLVM: alloca <4 x i16>, align 8
  %int16_t4_Val = alloca <4 x i16>, align 8

; CHECK-DAG: Variable [[#ptr_Function_v2int]] [[#]] 7
; CHECK-LLVM: alloca <2 x i32>, align 8
  %int2_Val = alloca <2 x i32>, align 8

; CHECK-DAG: Variable [[#ptr_Function_v3int]] [[#]] 7
; CHECK-LLVM: alloca <3 x i32>, align 16
  %int3_Val = alloca <3 x i32>, align 16

; CHECK-DAG: Variable [[#ptr_Function_v4int]] [[#]] 7
; CHECK-LLVM: alloca <4 x i32>, align 16
  %int4_Val = alloca <4 x i32>, align 16

; CHECK-DAG: Variable [[#ptr_Function_v2long]] [[#]] 7
; CHECK-LLVM: alloca <2 x i64>, align 16
  %int64_t2_Val = alloca <2 x i64>, align 16

; CHECK-DAG: Variable [[#ptr_Function_v3long]] [[#]] 7
; CHECK-LLVM: alloca <3 x i64>, align 32
  %int64_t3_Val = alloca <3 x i64>, align 32

; CHECK-DAG: Variable [[#ptr_Function_v4long]] [[#]] 7
; CHECK-LLVM: alloca <4 x i64>, align 32
  %int64_t4_Val = alloca <4 x i64>, align 32

  ret void
}
