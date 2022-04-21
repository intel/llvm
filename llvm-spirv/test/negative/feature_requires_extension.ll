; Check whether the translator reports an error for a module with token type
; if SPV_INTEL_token_type extension is not used.

; RUN: llvm-as < %s -o %t.bc
; RUN: not llvm-spirv %t.bc 2>&1 --spirv-allow-unknown-intrinsics | FileCheck %s

; CHECK: RequiresExtension: Feature requires the following SPIR-V extension:
; CHECK-NEXT: SPV_INTEL_token_type
; CHECK-NEXT: NOTE: LLVM module contains token type, which doesn't have analogs
; CHECK-SAME: in SPIR-V without extensions

; ModuleID = 'token.bc'
target datalayout = "e-p:32:32:32-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v16:16:16-v24:32:32-v32:32:32-v48:64:64-v64:64:64-v96:128:128-v128:128:128-v192:256:256-v256:256:256-v512:512:512-v1024:1024:1024"
target triple = "spir-unknown-unknown"

declare token @llvm.tokenfoo()

; Function Attrs: nounwind
define spir_kernel void @foo() #0 !kernel_arg_addr_space !2 !kernel_arg_access_qual !2 !kernel_arg_type !2 !kernel_arg_type_qual !2 !kernel_arg_base_type !2 {
entry:
  %tok = call token @llvm.tokenfoo()
  ret void
}

attributes #0 = { nounwind }

!spirv.MemoryModel = !{!0}
!opencl.enable.FP_CONTRACT = !{}
!spirv.Source = !{!1}
!opencl.spir.version = !{!0}
!opencl.ocl.version = !{!0}
!opencl.used.extensions = !{!2}
!opencl.used.optional.core.features = !{!2}
!spirv.Generator = !{!3}

!0 = !{i32 1, i32 2}
!1 = !{i32 3, i32 102000}
!2 = !{}
!3 = !{i16 6, i16 14}
