; RUN: llvm-as %s -o %t.bc
; RUN: llvm-spirv %t.bc -spirv-text -o - | FileCheck %s --check-prefix=CHECK-SPIRV
; RUN: llvm-spirv %t.bc -o %t.spv
; RUN: spirv-val %t.spv
; RUN: llvm-spirv -r %t.spv -o %t.rev.bc
; RUN: llvm-dis < %t.rev.bc | FileCheck %s --check-prefix=CHECK-LLVM

target datalayout = "e-p:32:32-i64:64-v16:16-v24:32-v32:32-v48:64-v96:128-v192:256-v256:256-v512:512-v1024:1024"
target triple = "spir"

; CHECK-SPIRV-DAG:  Name [[#SCALARi32:]] "select_i32"
; CHECK-SPIRV-DAG:  Name [[#SCALARPTR:]] "select_ptr"
; CHECK-SPIRV-DAG:  Name [[#VEC2i32:]] "select_i32v2"
; CHECK-SPIRV-DAG:  Name [[#VEC2i32v2:]] "select_v2i32v2"

; CHECK-SPIRV:      Function [[#]] [[#SCALARi32]]
; CHECK-SPIRV: FunctionParameter [[#]] [[#C:]]
; CHECK-SPIRV: FunctionParameter [[#]] [[#T:]]
; CHECK-SPIRV: FunctionParameter [[#]] [[#F:]]
; CHECK-SPIRV:      Select [[#]] [[#R:]] [[#C:]] [[#T]] [[#F]]
; CHECK-SPIRV:      ReturnValue [[#R]]
; CHECK-SPIRV: FunctionEnd

; CHECK-LLVM: @select_i32
; CHECK-LLVM: %[[R:.*]] = select i1 %{{.*}}, i32 %{{.*}}, i32 %{{.*}}
; CHECK-LLVM: ret i32 %[[R]]

define i32 @select_i32(i1 %c, i32 %t, i32 %f) {
  %r = select i1 %c, i32 %t, i32 %f
  ret i32 %r
}

; CHECK-SPIRV:      Function [[#]] [[#SCALARPTR]]
; CHECK-SPIRV: FunctionParameter [[#]] [[#C:]]
; CHECK-SPIRV: FunctionParameter [[#]] [[#T:]]
; CHECK-SPIRV: FunctionParameter [[#]] [[#F:]]
; CHECK-SPIRV:      Select [[#]] [[#R:]] [[#C:]] [[#T]] [[#F]]
; CHECK-SPIRV:      ReturnValue [[#R]]
; CHECK-SPIRV: FunctionEnd

; CHECK-LLVM: @select_ptr
; CHECK-LLVM: %[[R:.*]] = select i1 %{{.*}}, ptr %{{.*}}, ptr %{{.*}}
; CHECK-LLVM: ret ptr %[[R]]

define ptr @select_ptr(i1 %c, ptr %t, ptr %f) {
  %r = select i1 %c, ptr %t, ptr %f
  ret ptr %r
}

; CHECK-SPIRV:      Function [[#]] [[#VEC2i32]]
; CHECK-SPIRV: FunctionParameter [[#]] [[#C:]]
; CHECK-SPIRV: FunctionParameter [[#]] [[#T:]]
; CHECK-SPIRV: FunctionParameter [[#]] [[#F:]]
; CHECK-SPIRV:      Select [[#]] [[#R:]] [[#C:]] [[#T]] [[#F]]
; CHECK-SPIRV:      ReturnValue [[#R]]
; CHECK-SPIRV: FunctionEnd

; CHECK-LLVM: @select_i32v2
; CHECK-LLVM: %[[R:.*]] = select <2 x i1> %{{.*}}, <2 x i32> %{{.*}}, <2 x i32> %{{.*}}
; CHECK-LLVM: ret <2 x i32> %[[R]]

define <2 x i32> @select_i32v2(<2 x i1> %c, <2 x i32> %t, <2 x i32> %f) {
  %r = select <2 x i1> %c, <2 x i32> %t, <2 x i32> %f
  ret <2 x i32> %r
}

; CHECK-SPIRV:      Function [[#]] [[#VEC2i32v2]]
; CHECK-SPIRV: FunctionParameter [[#]] [[#C:]]
; CHECK-SPIRV: FunctionParameter [[#]] [[#T:]]
; CHECK-SPIRV: FunctionParameter [[#]] [[#F:]]
; CHECK-SPIRV:      Select [[#]] [[#R:]] [[#C:]] [[#T]] [[#F]]
; CHECK-SPIRV:      ReturnValue [[#R]]
; CHECK-SPIRV: FunctionEnd

; CHECK-LLVM: @select_v2i32v2
; CHECK-LLVM: %[[R:.*]] = select <2 x i1> %{{.*}}, <2 x i32> %{{.*}}, <2 x i32> %{{.*}}
; CHECK-LLVM: ret <2 x i32> %[[R]]

define <2 x i32> @select_v2i32v2(<2 x i1> %c, <2 x i32> %t, <2 x i32> %f) {
  %r = select <2 x i1> %c, <2 x i32> %t, <2 x i32> %f
  ret <2 x i32> %r
}
