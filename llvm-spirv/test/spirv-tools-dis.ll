; RUN: llvm-as %s -o %t.bc
; RUN: llvm-spirv %t.bc --spirv-tools-dis -o - | FileCheck %s
; RUN: llvm-spirv %t.bc --spirv-tools-dis -o - | spirv-as

; Verify that the --spirv-tools-dis options results in SPIRV-Tools compatible assembly.

; REQUIRES: libspirv_dis, spirv-as

; CHECK: %1 = OpExtInstImport "OpenCL.std"
; CHECK: %uint = OpTypeInt 32 0

target datalayout = "e-p:32:32-i64:64-v16:16-v24:32-v32:32-v48:64-v96:128-v192:256-v256:256-v512:512-v1024:1024"
target triple = "spir-unknown-unknown"

; Function Attrs: nounwind
define spir_kernel void @foo(i32 addrspace(1)* %a) !kernel_arg_addr_space !1 !kernel_arg_access_qual !2 !kernel_arg_type !3 !kernel_arg_base_type !4 !kernel_arg_type_qual !5 {
entry:
  %a.addr = alloca i32 addrspace(1)*, align 4
  store i32 addrspace(1)* %a, i32 addrspace(1)** %a.addr, align 4
  %0 = load i32 addrspace(1)*, i32 addrspace(1)** %a.addr, align 4
  store i32 0, i32 addrspace(1)* %0, align 4
  ret void
}

!opencl.enable.FP_CONTRACT = !{}
!opencl.spir.version = !{!6}
!opencl.ocl.version = !{!6}
!opencl.used.extensions = !{!7}
!opencl.used.optional.core.features = !{!7}
!opencl.compiler.options = !{!7}

!1 = !{i32 1}
!2 = !{!"none"}
!3 = !{!"int*"}
!4 = !{!"int*"}
!5 = !{!""}
!6 = !{i32 1, i32 2}
!7 = !{}
