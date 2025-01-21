; RUN: llvm-as %s -o %t.bc
; RUN: llvm-spirv %t.bc -o %t.spv --spirv-fp-contract=on
; RUN: llvm-spirv %t.spv -to-text -o - | FileCheck %s
; RUN: spirv-val %t.spv

; CHECK: EntryPoint 6 [[K1:[0-9]+]] "kernel_off_1"
; CHECK: EntryPoint 6 [[K2:[0-9]+]] "kernel_off_2"
; CHECK: EntryPoint 6 [[K3:[0-9]+]] "kernel_off_3"
; CHECK: EntryPoint 6 [[K4:[0-9]+]] "kernel_off_4"
; CHECK: EntryPoint 6 [[K5:[0-9]+]] "kernel_off_5"
; CHECK: EntryPoint 6 [[K6:[0-9]+]] "kernel_off_6"
; CHECK: EntryPoint 6 [[K7:[0-9]+]] "kernel_on_7"

; CHECK: ExecutionMode [[K1]] 31
; CHECK: ExecutionMode [[K2]] 31
; CHECK: ExecutionMode [[K3]] 31
; CHECK: ExecutionMode [[K4]] 31
; CHECK: ExecutionMode [[K5]] 31
; CHECK: ExecutionMode [[K6]] 31
; CHECK-NOT: ExecutionMode [[K7]] 31

target datalayout = "e-i64:64-v16:16-v24:32-v32:32-v48:64-v96:128-v192:256-v256:256-v512:512-v1024:1024"
target triple = "spir64"

define float @func_nested_off(float %a, float %b, float %c) {
entry:
  %0 = call float @func_off(float %a, float %b, float %c)
  ret float %0
}

define float @func_off(float %a, float %b, float %c) {
entry:
  %0 = call float @llvm.fmuladd.f32(float %a, float %b, float %c)
  %mul = fmul float %0, %b
  %add = fadd float %mul, %c
  ret float %add
}

declare float @llvm.fmuladd.f32(float, float, float) #0

define float @func_on(float %a, float %b, float %c) {
entry:
  %0 = call float @llvm.fmuladd.f32(float %a, float %b, float %c)
  ret float %0
}

define float @func_mul(float %a, float %b) {
entry:
  %0 = fmul float %a, %b
  ret float %0
}

declare float @func_extern(float, float, float)

define spir_kernel void @kernel_off_1(float %a, float %b, float %c) !kernel_arg_addr_space !3 !kernel_arg_access_qual !4 !kernel_arg_type !5 !kernel_arg_base_type !5 !kernel_arg_type_qual !6 {
entry:
  %mul = fmul float %a, %b
  %add = fadd float %mul, %c
  ret void
}

define spir_kernel void @kernel_off_2(float %a, float %b, float %c) !kernel_arg_addr_space !3 !kernel_arg_access_qual !4 !kernel_arg_type !5 !kernel_arg_base_type !5 !kernel_arg_type_qual !6 {
entry:
  %0 = call float @llvm.fmuladd.f32(float %a, float %b, float %c)
  %call = call float @func_off(float %0, float %b, float %c)
  ret void
}

define spir_kernel void @kernel_off_3(float %a, float %b, float %c) !kernel_arg_addr_space !3 !kernel_arg_access_qual !4 !kernel_arg_type !5 !kernel_arg_base_type !5 !kernel_arg_type_qual !6 {
entry:
  %call = call float @func_nested_off(float %a, float %b, float %c)
  ret void
}

define spir_kernel void @kernel_off_4(float %a, float %b, float %c) !kernel_arg_addr_space !3 !kernel_arg_access_qual !4 !kernel_arg_type !5 !kernel_arg_base_type !5 !kernel_arg_type_qual !6 {
entry:
  %0 = call float @llvm.fmuladd.f32(float %a, float %b, float %c)
  %call = call float @func_extern(float %0, float %b, float %c)
  ret void
}

define spir_kernel void @kernel_off_5(float %a, float %b, float %c) !kernel_arg_addr_space !3 !kernel_arg_access_qual !4 !kernel_arg_type !5 !kernel_arg_base_type !5 !kernel_arg_type_qual !6 {
entry:
  %mul = call float @func_off(float %a, float %b, float %c)
  %add = fadd contract float %mul, %c
  ret void
}

define spir_kernel void @kernel_off_6(float %a, float %b, float %c) !kernel_arg_addr_space !3 !kernel_arg_access_qual !4 !kernel_arg_type !5 !kernel_arg_base_type !5 !kernel_arg_type_qual !6 {
entry:
  %mul = call float @func_mul(float %a, float %b)
  %add = fadd float %mul, %c
  ret void
}

define spir_kernel void @kernel_on_7(float %a, float %b, float %c) !kernel_arg_addr_space !3 !kernel_arg_access_qual !4 !kernel_arg_type !5 !kernel_arg_base_type !5 !kernel_arg_type_qual !6 {
entry:
  %mul = call float @func_mul(float %a, float %b)
  %add = fadd contract float %mul, %c
  ret void
}

attributes #0 = { nounwind readnone speculatable willreturn }


!llvm.module.flags = !{!0}
!opencl.ocl.version = !{!1}
!llvm.ident = !{!2}

!0 = !{i32 1, !"wchar_size", i32 4}
!1 = !{i32 1, i32 0}
!2 = !{!"clang version 11.0.0"}
!3 = !{i32 0, i32 0, i32 0}
!4 = !{!"none", !"none", !"none"}
!5 = !{!"float", !"float", !"float"}
!6 = !{!"", !"", !""}
