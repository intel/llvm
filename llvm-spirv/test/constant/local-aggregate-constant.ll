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

%aggregate = type { i8, i32 }

define %aggregate @getConstant() {
  ret %aggregate { i8 1, i32 2 }
}

; CHECK:     Name [[#GET:]] "getConstant"
; CHECK-DAG: TypeInt [[#I8:]] 8 0
; CHECK-DAG: TypeInt [[#I32:]] 32 0
; CHECK-DAG: Constant [[#I8]] [[#CST_I8:]] 1
; CHECK-DAG: Constant [[#I32]] [[#CST_I32:]] 2
; CHECK-DAG: TypeStruct [[#AGGREGATE:]] [[#I8]] [[#I32]]
; CHECK-DAG: ConstantComposite [[#AGGREGATE]] [[#CST_AGGREGATE:]] [[#CST_I8]] [[#CST_I32]]

; CHECK: Function [[#AGGREGATE]] [[#GET]]
; CHECK: ReturnValue [[#CST_AGGREGATE]]
; CHECK: FunctionEnd

; CHECK-LLVM: ret %{{.*}} { i8 1, i32 2 }
