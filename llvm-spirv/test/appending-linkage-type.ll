; RUN: llvm-as < %s -o %t.bc
; RUN: llvm-spirv %t.bc -o %t.spv -spirv-ext=+SPV_INTEL_vector_compute,+SPV_INTEL_usm_storage_classes
; RUN: llvm-spirv -r %t.spv -o %t.rev.bc
; RUN: llvm-dis %t.rev.bc -o %t.rev.ll
; RUN: FileCheck %s --input-file %t.rev.ll --check-prefix CHECK-LLVM

; ModuleID = 'appending-linkage-type'
target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v16:16:16-v24:32:32-v32:32:32-v48:64:64-v64:64:64-v96:128:128-v128:128:128-v192:256:256-v256:256:256-v512:512:512-v1024:1024:1024"
target triple = "spir64-unknown-unknown"

%union.int_double = type { i64 }

@_c0 = internal addrspace(1) constant %union.int_double { i64 4607182418800017408 }, align 8 #0
@._c0.ref = internal constant %union.int_double addrspace(1)* @_c0 #0
; CHECK-LLVM: @llvm.compiler.used = appending global
@llvm.compiler.used = appending global [1 x i8*] [i8* bitcast (%union.int_double addrspace(1)** @._c0.ref to i8*)] #0

attributes #0 = { "VCGlobalVariable" }

!spirv.MemoryModel = !{!0}
!opencl.enable.FP_CONTRACT = !{}
!spirv.Source = !{!1}
!opencl.spir.version = !{!2}
!opencl.ocl.version = !{!1}
!opencl.used.extensions = !{!3}
!opencl.used.optional.core.features = !{!3}
!spirv.Generator = !{!4}

!0 = !{i32 2, i32 2}
!1 = !{i32 0, i32 0}
!2 = !{i32 1, i32 2}
!3 = !{}
!4 = !{i16 6, i16 14}
