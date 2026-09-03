; RUN: llvm-as %s -o %t.bc
; RUN: not llvm-spirv -s %t.bc 2>&1 | FileCheck %s --check-prefix=CHECK-ERROR
; RUN: llvm-spirv --spirv-ext=+SPV_EXT_long_vector %t.bc -o %t.spv
; RUN: llvm-spirv %t.spv -to-text -o - | FileCheck %s

; CHECK-ERROR: Unsupported vector type with 20 elements

; CHECK-DAG: Capability LongVectorEXT
; CHECK-DAG: Extension "SPV_EXT_long_vector"
; CHECK-DAG: TypeInt [[#I32:]] 32 0
; CHECK-DAG: TypeInt [[#I8:]] 8 0
; CHECK-DAG: TypeVector [[#]] [[#I32]] 1
; CHECK-DAG: TypeVector [[#]] [[#I8]] 4
; CHECK-DAG: TypeVector [[#]] [[#I32]] 5
; CHECK-DAG: TypeVector [[#]] [[#I8]] 20

; ModuleID = 'lower-non-standard-vec-with-ext'
target datalayout = "e-i64:64-v16:16-v24:32-v32:32-v48:64-v96:128-v192:256-v256:256-v512:512-v1024:1024-n8:16:32:64"
target triple = "spir64-unknown-unknown"

; Function Attrs: convergent norecurse
define dso_local spir_func void @vmult2() local_unnamed_addr {
entry:
  %0 = bitcast <1 x i32> <i32 65793> to <4 x i8>
  %1 = extractelement <4 x i8> %0, i32 0
  %2 = bitcast <1 x i32> <i32 131586> to <4 x i8>
  %3 = extractelement <4 x i8> %2, i32 0
  %4 = bitcast <5 x i32> <i32 1, i32 1, i32 1, i32 1, i32 1> to <20 x i8>
  ret void
}
