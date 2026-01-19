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

; CHECK-DAG: Capability Float16Buffer
; CHECK-DAG: Capability Float64

; CHECK-DAG: TypeFloat [[#half:]] 16
; CHECK-DAG: TypeFloat [[#float:]] 32
; CHECK-DAG: TypeFloat [[#double:]] 64

; CHECK-DAG: TypeVector [[#v2half:]] [[#half]] 2
; CHECK-DAG: TypeVector [[#v3half:]] [[#half]] 3
; CHECK-DAG: TypeVector [[#v4half:]] [[#half]] 4
; CHECK-DAG: TypeVector [[#v2float:]] [[#float]] 2
; CHECK-DAG: TypeVector [[#v3float:]] [[#float]] 3
; CHECK-DAG: TypeVector [[#v4float:]] [[#float]] 4
; CHECK-DAG: TypeVector [[#v2double:]] [[#double]] 2
; CHECK-DAG: TypeVector [[#v3double:]] [[#double]] 3
; CHECK-DAG: TypeVector [[#v4double:]] [[#double]] 4

; CHECK-DAG: TypePointer [[#ptr_Function_half:]] 7 [[#half]]
; CHECK-DAG: TypePointer [[#ptr_Function_float:]] 7 [[#float]]
; CHECK-DAG: TypePointer [[#ptr_Function_double:]] 7 [[#double]]
; CHECK-DAG: TypePointer [[#ptr_Function_v2half:]] 7 [[#v2half]]
; CHECK-DAG: TypePointer [[#ptr_Function_v3half:]] 7 [[#v3half]]
; CHECK-DAG: TypePointer [[#ptr_Function_v4half:]] 7 [[#v4half]]
; CHECK-DAG: TypePointer [[#ptr_Function_v2float:]] 7 [[#v2float]]
; CHECK-DAG: TypePointer [[#ptr_Function_v3float:]] 7 [[#v3float]]
; CHECK-DAG: TypePointer [[#ptr_Function_v4float:]] 7 [[#v4float]]
; CHECK-DAG: TypePointer [[#ptr_Function_v2double:]] 7 [[#v2double]]
; CHECK-DAG: TypePointer [[#ptr_Function_v3double:]] 7 [[#v3double]]
; CHECK-DAG: TypePointer [[#ptr_Function_v4double:]] 7 [[#v4double]]

; CHECK-DAG: Variable [[#ptr_Function_half]] [[#]] 7
; CHECK-LLVM: alloca half, align 2
  %half_Val = alloca half, align 2

; CHECK-DAG: Variable [[#ptr_Function_float]] [[#]] 7
; CHECK-LLVM: alloca float, align 4
  %float_Val = alloca float, align 4

; CHECK-DAG: Variable [[#ptr_Function_double]] [[#]] 7
; CHECK-LLVM: alloca double, align 8
  %double_Val = alloca double, align 8

; CHECK-DAG: Variable [[#ptr_Function_v2half]] [[#]] 7
; CHECK-LLVM: alloca <2 x half>, align 4
  %half2_Val = alloca <2 x half>, align 4

; CHECK-DAG: Variable [[#ptr_Function_v3half]] [[#]] 7
; CHECK-LLVM: alloca <3 x half>, align 8
  %half3_Val = alloca <3 x half>, align 8

; CHECK-DAG: Variable [[#ptr_Function_v4half]] [[#]] 7
; CHECK-LLVM: alloca <4 x half>, align 8
  %half4_Val = alloca <4 x half>, align 8

; CHECK-DAG: Variable [[#ptr_Function_v2float]] [[#]] 7
; CHECK-LLVM: alloca <2 x float>, align 8
  %float2_Val = alloca <2 x float>, align 8

; CHECK-DAG: Variable [[#ptr_Function_v3float]] [[#]] 7
; CHECK-LLVM: alloca <3 x float>, align 16
  %float3_Val = alloca <3 x float>, align 16

; CHECK-DAG: Variable [[#ptr_Function_v4float]] [[#]] 7
; CHECK-LLVM: alloca <4 x float>, align 16
  %float4_Val = alloca <4 x float>, align 16

; CHECK-DAG: Variable [[#ptr_Function_v2double]] [[#]] 7
; CHECK-LLVM: alloca <2 x double>, align 16
  %double2_Val = alloca <2 x double>, align 16

; CHECK-DAG: Variable [[#ptr_Function_v3double]] [[#]] 7
; CHECK-LLVM: alloca <3 x double>, align 32
  %double3_Val = alloca <3 x double>, align 32

; CHECK-DAG: Variable [[#ptr_Function_v4double]] [[#]] 7
; CHECK-LLVM: alloca <4 x double>, align 32
  %double4_Val = alloca <4 x double>, align 32
  ret void
}
