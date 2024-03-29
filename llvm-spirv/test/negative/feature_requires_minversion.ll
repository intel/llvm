; Check whether the translator reports an error for a module that uses a
; construct only available in a SPIR-V version that's higher than what was
; requested using --spirv-max-version.

; RUN: llvm-as < %s -o %t.bc
; RUN: not llvm-spirv --spirv-max-version=1.0 %t.bc 2>&1 | FileCheck %s
; RUN: not llvm-spirv --spirv-max-version=1.1 %t.bc 2>&1 | FileCheck %s
; RUN: not llvm-spirv --spirv-max-version=1.2 %t.bc 2>&1 | FileCheck %s
; RUN: llvm-spirv --spirv-max-version=1.3 %t.bc -o %t.spv
; RUN: spirv-val %t.spv

; CHECK: RequiresVersion: Cannot fulfill SPIR-V version restriction:
; CHECK-NEXT: SPIR-V version was restricted to at most 1.{{[012]}} ([[#]]) but a construct from the input requires SPIR-V version 1.3 (66304) or above

; ModuleID = 'foo.bc'
target datalayout = "e-i64:64-v16:16-v24:32-v32:32-v48:64-v96:128-v192:256-v256:256-v512:512-v1024:1024"
target triple = "spir64"

; Function Attrs: convergent
declare dso_local spir_func <4 x i32> @_Z16sub_group_balloti(i32) local_unnamed_addr #1

; Function Attrs: nounwind
define dso_local spir_kernel void @testVersionReq() local_unnamed_addr #0 !kernel_arg_addr_space !3 !kernel_arg_access_qual !3 !kernel_arg_type !3 !kernel_arg_base_type !3 !kernel_arg_type_qual !3 {
entry:
  %1 = tail call spir_func <4 x i32> @_Z16sub_group_balloti(i32 0) #1
  ret void
}

attributes #0 = { nounwind }
attributes #1 = { convergent }

!spirv.MemoryModel = !{!0}
!opencl.enable.FP_CONTRACT = !{}
!spirv.Source = !{!1}
!opencl.spir.version = !{!0}
!opencl.ocl.version = !{!0}
!opencl.used.extensions = !{!2}
!opencl.used.optional.core.features = !{!2}
!spirv.Generator = !{!2}

!0 = !{i32 1, i32 2}
!1 = !{i32 3, i32 102000}
!2 = !{}
!3 = !{}
