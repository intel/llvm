; This is a basic test for Lowering ESIMD constructs after splitting.
; This test also implicitly checks that input module is not reused
; for ESIMD kernels in any case.

; No lowering
; RUN: sycl-post-link -properties -split-esimd -S < %s -o %t.table
; RUN: FileCheck %s -input-file=%t_esimd_0.ll --check-prefixes CHECK-NO-LOWERING

; Default lowering
; RUN: sycl-post-link -properties -split-esimd -lower-esimd -S < %s -o %t.table
; RUN: FileCheck %s -input-file=%t_esimd_0.ll --check-prefixes CHECK-O2

; -O2 lowering
; RUN: sycl-post-link -properties -split-esimd -lower-esimd -O2 -S < %s -o %t.table
; RUN: FileCheck %s -input-file=%t_esimd_0.ll --check-prefixes CHECK-O2

; -O0 lowering
; RUN: sycl-post-link -properties -split-esimd -lower-esimd -O0 -S < %s -o %t.table
; RUN: FileCheck %s -input-file=%t_esimd_0.ll --check-prefixes CHECK-O0

target datalayout = "e-i64:64-v16:16-v24:32-v32:32-v48:64-v96:128-v192:256-v256:256-v512:512-v1024:1024"
target triple = "spir64-unknown-linux"

@__spirv_BuiltInGlobalInvocationId = external dso_local local_unnamed_addr addrspace(1) constant <3 x i64>, align 32

declare dso_local spir_func void @_Z13__esimd_fenceh(i8 noundef zeroext)
declare dso_local spir_func void @_Z15__esimd_barrierv()

define dso_local spir_kernel void @ESIMD_kernel() #0 !sycl_explicit_simd !3 {
entry:
  %0 = load <3 x i64>, <3 x i64> addrspace(4)* addrspacecast (<3 x i64> addrspace(1)* @__spirv_BuiltInGlobalInvocationId to <3 x i64> addrspace(4)*), align 32
  %1 = extractelement <3 x i64> %0, i64 0
  call spir_func void @_Z15__esimd_barrierv()
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

; By default, no lowering is performed
; CHECK-NO-LOWERING: define dso_local spir_kernel void @ESIMD_kernel()
; CHECK-NO-LOWERING: entry:
; CHECK-NO-LOWERING:   %0 = load <3 x i64>, {{.*}} addrspacecast {{.*}} @__spirv_BuiltInGlobalInvocationId
; CHECK-NO-LOWERING:   %1 = extractelement <3 x i64> %0, i64 0
; CHECK-NO-LOWERING:   call spir_func void @_Z15__esimd_barrierv()
; CHECK-NO-LOWERING:   ret void
; CHECK-NO-LOWERING: }

; With -O0, we only lower ESIMD code, but no other optimizations
; CHECK-O0: define dso_local spir_kernel void @ESIMD_kernel() #{{[0-9]}} !sycl_explicit_simd !3 !intel_reqd_sub_group_size !4 {
; CHECK-O0: entry:
; CHECK-O0:   %0 = load <3 x i64>, {{.*}} addrspacecast {{.*}} @__spirv_BuiltInGlobalInvocationId
; CHECK-O0:   %1 = extractelement <3 x i64> %0, i64 0
; CHECK-O0:   call void @llvm.genx.barrier()
; CHECK-O0:   ret void
; CHECK-O0: }

; With -O2, unused call was optimized away
; CHECK-O2: define dso_local spir_kernel void @ESIMD_kernel()
; CHECK-O2: entry:
; CHECK-O2:   call void @llvm.genx.barrier()
; CHECK-O2:   ret void
; CHECK-O2: }
