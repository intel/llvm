; RUN: sycl-post-link -split-esimd -lower-esimd -O2 -S %s -o %t.table
; RUN: FileCheck %s -input-file=%t_esimd_0.ll
; This test checks that IR code below can be successfully processed by
; sycl-post-link. In this IR no extractelement instruction and no casting are used

target datalayout = "e-i64:64-v16:16-v24:32-v32:32-v48:64-v96:128-v192:256-v256:256-v512:512-v1024:1024-n8:16:32:64"
target triple = "spir64-unknown-unknown"

@__spirv_BuiltInGlobalInvocationId = external dso_local local_unnamed_addr addrspace(1) constant <3 x i64>, align 32

; Function Attrs: convergent norecurse
define dso_local spir_kernel void @ESIMD_kernel() #0 !sycl_explicit_simd !3 {
entry:
  %0 = load i64, i64 addrspace(1)* getelementptr (<3 x i64>, <3 x i64> addrspace(1)* @__spirv_BuiltInGlobalInvocationId, i64 0, i64 0), align 32
  %conv = trunc i64 %0 to i32
  ret void
}

attributes #0 = { "sycl-module-id"="a.cpp" }

!llvm.module.flags = !{!0}
!opencl.spir.version = !{!1}
!spirv.Source = !{!2}

!0 = !{i32 1, !"wchar_size", i32 4}
!1 = !{i32 1, i32 2}
!2 = !{i32 0, i32 100000}
!3 = !{}

; CHECK: define dso_local spir_kernel void @ESIMD_kernel()
; CHECK:   call <3 x i32> @llvm.genx.local.id.v3i32()
; CHECK:   call <3 x i32> @llvm.genx.local.size.v3i32()
; CHECK:   call i32 @llvm.genx.group.id.x()
; CHECK:   ret void
; CHECK: }
