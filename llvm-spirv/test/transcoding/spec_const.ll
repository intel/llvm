; RUN: llvm-as < %s -o %t.bc
; RUN: llvm-spirv %t.bc -o %t.spv
; RUN: spirv-val %t.spv
; RUN: llvm-spirv %t.spv -to-text -o - | FileCheck %s --check-prefix=CHECK-SPIRV
; RUN: llvm-spirv -r %t.spv -o %t.rev.bc
; RUN: llvm-dis < %t.rev.bc | FileCheck %s --check-prefix=CHECK-LLVM
; RUN: llvm-spirv -r -spec-const "0:i1:1 1:i8:11 2:i16:22 3:i32:33 4:i64:44 5:f16:5.5 6:f32:6.6 7:f64:7.7" %t.spv -o %t.rev.spec.bc
; RUN: llvm-dis < %t.rev.spec.bc | FileCheck %s --check-prefix=CHECK-LLVM-SPEC

; CHECK-SPIRV-NOT: Capability Matrix
; CHECK-SPIRV-NOT: Capability Shader
; CHECK-SPIRV: Capability Kernel

; CHECK-SPIRV-DAG: Decorate [[SC0:[0-9]+]] SpecId 0
; CHECK-SPIRV-DAG: Decorate [[SC1:[0-9]+]] SpecId 1
; CHECK-SPIRV-DAG: Decorate [[SC2:[0-9]+]] SpecId 2
; CHECK-SPIRV-DAG: Decorate [[SC3:[0-9]+]] SpecId 3
; CHECK-SPIRV-DAG: Decorate [[SC4:[0-9]+]] SpecId 4
; CHECK-SPIRV-DAG: Decorate [[SC5:[0-9]+]] SpecId 5
; CHECK-SPIRV-DAG: Decorate [[SC6:[0-9]+]] SpecId 6
; CHECK-SPIRV-DAG: Decorate [[SC7:[0-9]+]] SpecId 7

; CHECK-SPIRV-DAG: SpecConstantFalse {{[0-9]+}} [[SC0]]
; CHECK-SPIRV-DAG: SpecConstant {{[0-9]+}} [[SC1]] 100
; CHECK-SPIRV-DAG: SpecConstant {{[0-9]+}} [[SC2]] 1
; CHECK-SPIRV-DAG: SpecConstant {{[0-9]+}} [[SC3]] 2
; CHECK-SPIRV-DAG: SpecConstant {{[0-9]+}} [[SC4]] 3 0
; CHECK-SPIRV-DAG: SpecConstant {{[0-9]+}} [[SC5]] 14336
; CHECK-SPIRV-DAG: SpecConstant {{[0-9]+}} [[SC6]] 1067450368
; CHECK-SPIRV-DAG: SpecConstant {{[0-9]+}} [[SC7]] 0 1073807360

target datalayout = "e-p:32:32-i64:64-v16:16-v24:32-v32:32-v48:64-v96:128-v192:256-v256:256-v512:512-v1024:1024"
target triple = "spir"
; Function Attrs: nofree norecurse nounwind writeonly
 define spir_kernel void @foo(i8 addrspace(1)* nocapture %b, i8 addrspace(1)* nocapture %c, i16 addrspace(1)* nocapture %s, i32 addrspace(1)* nocapture %i, i64 addrspace(1)* nocapture %l, half addrspace(1)* nocapture %h, float addrspace(1)* nocapture %f, double addrspace(1)* nocapture %d) local_unnamed_addr #0 !kernel_arg_addr_space !3 !kernel_arg_access_qual !4 !kernel_arg_type !5 !kernel_arg_base_type !5 !kernel_arg_type_qual !6 {
entry:
  ; CHECK-LLVM: store i8 0, i8 addrspace(1)* %b, align 1
  ; CHECK-LLVM-SPEC: store i8 1, i8 addrspace(1)* %b, align 1
  %0 = call i1 @_Z20__spirv_SpecConstantib(i32 0, i1 false)
  %conv = zext i1 %0 to i8
  store i8 %conv, i8 addrspace(1)* %b, align 1

  ; CHECK-LLVM: store i8 100, i8 addrspace(1)* %c, align 1
  ; CHECK-LLVM-SPEC: store i8 11, i8 addrspace(1)* %c, align 1
  %1 = call i8 @_Z20__spirv_SpecConstantia(i32 1, i8 100)
  store i8 %1, i8 addrspace(1)* %c, align 1

  ; CHECK-LLVM: store i16 1, i16 addrspace(1)* %s, align 2
  ; CHECK-LLVM-SPEC: store i16 22, i16 addrspace(1)* %s, align 2
  %2 = call i16 @_Z20__spirv_SpecConstantis(i32 2, i16 1)
  store i16 %2, i16 addrspace(1)* %s, align 2

  ; CHECK-LLVM: store i32 2, i32 addrspace(1)* %i, align 4
  ; CHECK-LLVM-SPEC: store i32 33, i32 addrspace(1)* %i, align 4
  %3 = call i32 @_Z20__spirv_SpecConstantii(i32 3, i32 2)
  store i32 %3, i32 addrspace(1)* %i, align 4

  ; CHECK-LLVM: store i64 3, i64 addrspace(1)* %l, align 8
  ; CHECK-LLVM-SPEC: store i64 44, i64 addrspace(1)* %l, align 8
  %4 = call i64 @_Z20__spirv_SpecConstantix(i32 4, i64 3)
  store i64 %4, i64 addrspace(1)* %l, align 8

  ; CHECK-LLVM: store half 0xH3800, half addrspace(1)* %h, align 2
  ; CHECK-LLVM-SPEC: store half 0xH4580, half addrspace(1)* %h, align 2
  %5 = call half @_Z20__spirv_SpecConstantih(i32 5, half 0xH3800)
  store half %5, half addrspace(1)* %h, align 2

  ; CHECK-LLVM: store float 1.250000e+00, float addrspace(1)* %f, align 4
  ; CHECK-LLVM-SPEC: store float 0x401A666660000000, float addrspace(1)* %f, align 4
  %6 = call float @_Z20__spirv_SpecConstantif(i32 6, float 1.250000e+00)
  store float %6, float addrspace(1)* %f, align 4

  ; CHECK-LLVM: store double 2.125000e+00, double addrspace(1)* %d, align 8
  ; CHECK-LLVM-SPEC: store double 7.700000e+00, double addrspace(1)* %d, align 8
  %7 = call double @_Z20__spirv_SpecConstantid(i32 7, double 2.125000e+00)
  store double %7, double addrspace(1)* %d, align 8
  ret void
}

declare i1 @_Z20__spirv_SpecConstantib(i32, i1)
declare i8 @_Z20__spirv_SpecConstantia(i32, i8)
declare i16 @_Z20__spirv_SpecConstantis(i32, i16)
declare i32 @_Z20__spirv_SpecConstantii(i32, i32)
declare i64 @_Z20__spirv_SpecConstantix(i32, i64)
declare half @_Z20__spirv_SpecConstantih(i32, half)
declare float @_Z20__spirv_SpecConstantif(i32, float)
declare double @_Z20__spirv_SpecConstantid(i32, double)

attributes #0 = { nofree norecurse nounwind writeonly "correctly-rounded-divide-sqrt-fp-math"="false" "disable-tail-calls"="false" "frame-pointer"="none" "less-precise-fpmad"="false" "min-legal-vector-width"="0" "no-infs-fp-math"="false" "no-jump-tables"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "no-trapping-math"="false" "stack-protector-buffer-size"="8" "uniform-work-group-size"="false" "unsafe-fp-math"="false" "use-soft-float"="false" }

!llvm.module.flags = !{!0}
!opencl.ocl.version = !{!1}
!opencl.spir.version = !{!1}
!opencl.used.extensions = !{!7}
!llvm.ident = !{!2}

!0 = !{i32 1, !"wchar_size", i32 4}
!1 = !{i32 2, i32 0}
!2 = !{!"clang version 11.0.0 (https://github.com/llvm/llvm-project.git f11016b41a94d2bad8824d5e2833d141fda24502)"}
!3 = !{i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1}
!4 = !{!"none", !"none", !"none", !"none", !"none", !"none", !"none", !"none"}
!5 = !{!"bool*", !"char*", !"short*", !"int*", !"long*", !"half*", !"float*", !"double*"}
!6 = !{!"", !"", !"", !"", !"", !"", !"", !""}
!7 = !{!"cl_khr_fp16"}
