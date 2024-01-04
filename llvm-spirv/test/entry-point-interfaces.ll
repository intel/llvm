; RUN: llvm-as %s -o %t.bc

; RUN: llvm-spirv %t.bc -o %t.spv
; RUN: spirv-val --target-env spv1.4 %t.spv
; RUN: llvm-spirv -to-text %t.spv -o %t.from.spv.spt
; RUN: FileCheck < %t.from.spv.spt %s --check-prefix=CHECK-SPIRV

; RUN: llvm-spirv -spirv-text %t.bc -o %t.from.bc.spt
; RUN: FileCheck < %t.from.bc.spt %s --check-prefix=CHECK-SPIRV

; CHECK-SPIRV: 7 EntryPoint 6 [[#]] "test" [[#Interface1:]] [[#Interface2:]]
; CHECK-SPIRV: TypeInt [[#TypeInt:]] 32 0
; CHECK-SPIRV: Constant [[#TypeInt]] [[#Constant1:]] 1
; CHECK-SPIRV: Constant [[#TypeInt]] [[#Constant2:]] 3
; CHECK-SPIRV: Variable [[#]] [[#Interface1]] 0 [[#Constant1]]
; CHECK-SPIRV: Variable [[#]] [[#Interface2]] 0 [[#Constant2]]

; ModuleID = 'source.cpp'
source_filename = "source.cpp"
target datalayout = "e-p:32:32-i64:64-v16:16-v24:32-v32:32-v48:64-v96:128-v192:256-v256:256-v512:512-v1024:1024-n8:16:32:64"
target triple = "spir"

@var = dso_local addrspace(2) constant i32 1, align 4
@var2 = dso_local addrspace(2) constant i32 3, align 4
@var.const = private unnamed_addr addrspace(2) constant i32 1, align 4
@var2.const = private unnamed_addr addrspace(2) constant i32 3, align 4

; Function Attrs: convergent noinline norecurse nounwind optnone
define dso_local spir_kernel void @test() #0 !kernel_arg_addr_space !2 !kernel_arg_access_qual !2 !kernel_arg_type !2 !kernel_arg_base_type !2 !kernel_arg_type_qual !2 !kernel_arg_host_accessible !2 !kernel_arg_pipe_depth !2 !kernel_arg_pipe_io !2 !kernel_arg_buffer_location !2 {
entry:
  %0 = load i32, ptr addrspace(2) @var.const, align 4
  %1 = load i32, ptr addrspace(2) @var2.const, align 4
  %mul = mul nsw i32 %0, %1
  %mul1 = mul nsw i32 %mul, 2
  ret void
}

attributes #0 = { convergent noinline norecurse nounwind optnone "frame-pointer"="all" "min-legal-vector-width"="0" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "uniform-work-group-size"="false" }

!opencl.enable.FP_CONTRACT = !{}
!opencl.ocl.version = !{!0}
!opencl.spir.version = !{!0}
!llvm.module.flags = !{!1}
!opencl.used.extensions = !{!2}
!opencl.used.optional.core.features = !{!2}
!opencl.compiler.options = !{!2}
!llvm.ident = !{!3}

!0 = !{i32 2, i32 0}
!1 = !{i32 7, !"frame-pointer", i32 2}
!2 = !{}
!3 = !{!"Compiler"}
