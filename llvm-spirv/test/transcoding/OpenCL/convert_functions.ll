; This test checks that functions with `convert_` prefix are translated as
; OpenCL builtins only in case they match the specification. Otherwise, we
; expect such functions to be translated to SPIR-V FunctionCall.

; RUN: llvm-as %s -o %t.bc
; RUN: llvm-spirv %t.bc -o %t.spv
; RUN: spirv-val %t.spv
; RUN: llvm-spirv %t.spv -to-text -o %t.spt
; RUN: FileCheck < %t.spt %s -check-prefix=CHECK-SPIRV
; RUN: llvm-spirv %t.spv -r -o - | llvm-dis -o %t.rev.ll
; RUN: FileCheck < %t.rev.ll %s -check-prefix=CHECK-LLVM

; CHECK-SPIRV: Name [[#Func:]] "_Z18convert_float_func"
; CHECK-SPIRV: Name [[#Func1:]] "_Z20convert_uint_satfunc"
; CHECK-SPIRV: Name [[#Func2:]] "_Z21convert_float_rtzfunc"
; CHECK-SPIRV-DAG: TypeVoid [[#VoidTy:]]
; CHECK-SPIRV-DAG: TypeFloat [[#FloatTy:]] 32

; CHECK-SPIRV: Function [[#VoidTy]] [[#Func]]
; CHECK-SPIRV: ConvertSToF [[#FloatTy]] [[#ConvertId:]] [[#]]
; CHECK-SPIRV: FunctionCall [[#VoidTy]] [[#]] [[#Func]] [[#ConvertId]]
; CHECK-SPIRV: FunctionCall [[#VoidTy]] [[#]] [[#Func1]] [[#]]
; CHECK-SPIRV: FunctionCall [[#VoidTy]] [[#]] [[#Func2]] [[#ConvertId]]
; CHECK-SPIRV-NOT: FConvert
; CHECK-SPIRV-NOT: ConvertUToF

target datalayout = "e-p:32:32-i64:64-v16:16-v24:32-v32:32-v48:64-v96:128-v192:256-v256:256-v512:512-v1024:1024-n8:16:32:64"
target triple = "spir"

; Function Attrs: convergent noinline norecurse nounwind optnone
define dso_local spir_func void @_Z18convert_float_func(float noundef %x) #0 {
entry:
  %x.addr = alloca float, align 4
  store float %x, ptr %x.addr, align 4
  ret void
}

define dso_local spir_func void @_Z20convert_uint_satfunc(i32 noundef %x) #0 {
entry:
  ret void
}

define dso_local spir_func void @_Z21convert_float_rtzfunc(float noundef %x) #0 {
entry:
  ret void
}

; Function Attrs: convergent noinline norecurse nounwind optnone
define dso_local spir_func void @convert_int_bf16(i32 noundef %x) #0 {
entry:
  %x.addr = alloca i32, align 4
  store i32 %x, ptr %x.addr, align 4
  %0 = load i32, ptr %x.addr, align 4
; CHECK-LLVM: %[[Call:[a-z]+]] = sitofp i32 %[[#]] to float
; CHECK-LLVM: call spir_func void @_Z18convert_float_func(float %[[Call]])
; CHECK-LLVM: call spir_func void @_Z20convert_uint_satfunc(i32 %[[#]])
; CHECK-LLVM: call spir_func void @_Z21convert_float_rtzfunc(float %[[Call]])
  %call = call spir_func float @_Z13convert_floati(i32 noundef %0) #1
  call spir_func void @_Z18convert_float_func(float noundef %call) #0
  call spir_func void @_Z20convert_uint_satfunc(i32 noundef %0) #0
  call spir_func void @_Z21convert_float_rtzfunc(float noundef %call) #0
  ret void
}

; Function Attrs: convergent nounwind willreturn memory(none)
declare spir_func float @_Z13convert_floati(i32 noundef) #1

attributes #0 = { convergent nounwind }
attributes #1 = { convergent nounwind willreturn memory(none) }

!opencl.ocl.version = !{!0}
!opencl.spir.version = !{!0}

!0 = !{i32 3, i32 0}
