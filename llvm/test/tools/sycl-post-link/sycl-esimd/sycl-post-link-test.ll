; -O0 lowering, requires `-force-disable-esimd-opt` to disable all optimizations.
; RUN: sycl-post-link -properties -split-esimd -lower-esimd -O0 -force-disable-esimd-opt -S < %s -o %t.table
; RUN: FileCheck %s -input-file=%t_esimd_0.ll
; This test checks that IR code below can be successfully processed by
; sycl-post-link. In this IR no extractelement instruction and no casting are used

target datalayout = "e-i64:64-v16:16-v24:32-v32:32-v48:64-v96:128-v192:256-v256:256-v512:512-v1024:1024-n8:16:32:64"
target triple = "spir64-unknown-unknown"

; Function Attrs: convergent norecurse
define dso_local spir_kernel void @kernel_SubgroupLocalInvocationId(ptr addrspace(1) noundef align 8 %_arg_DoNotOptimize, ptr addrspace(1) noundef align 4 %_arg_DoNotOptimize32) #0 !sycl_explicit_simd !3 {
entry:
  %0 = call spir_func i32 @_Z40__spirv_BuiltInSubgroupLocalInvocationIdv()
  %conv.i = zext i32 %0 to i64
  store i64 %conv.i, ptr addrspace(1) %_arg_DoNotOptimize, align 8
  %add.i = add i32 %0, 3
  store i32 %add.i, ptr addrspace(1) %_arg_DoNotOptimize32, align 4
  ret void
}
; CHECK: %conv.i = zext i32 0 to i64
; CHECK: store i64 %conv.i, ptr addrspace(1) %_arg_DoNotOptimize, align 8
; CHECK: %add.i = add i32 0, 3
; CHECK: store i32 %add.i, ptr addrspace(1) %_arg_DoNotOptimize32, align 4

; Function Attrs: convergent norecurse
define dso_local spir_kernel void @kernel_SubgroupSize(ptr addrspace(1) noundef align 8 %_arg_DoNotOptimize, ptr addrspace(1) noundef align 4 %_arg_DoNotOptimize32)#0 !sycl_explicit_simd !3{
entry:
  %0 = call spir_func i32 @_Z27__spirv_BuiltInSubgroupSizev()
  %conv.i = zext i32 %0 to i64
  store i64 %conv.i, ptr addrspace(1) %_arg_DoNotOptimize, align 8
  %add.i = add i32 %0, 7
  store i32 %add.i, ptr addrspace(1) %_arg_DoNotOptimize32, align 4
  ret void
}
; CHECK: %conv.i = zext i32 1 to i64
; CHECK: store i64 %conv.i, ptr addrspace(1) %_arg_DoNotOptimize, align 8
; CHECK: %add.i = add i32 1, 7
; CHECK: store i32 %add.i, ptr addrspace(1) %_arg_DoNotOptimize32, align 4

; Function Attrs: convergent norecurse
define dso_local spir_kernel void @kernel_SubgroupMaxSize(ptr addrspace(1) noundef align 8 %_arg_DoNotOptimize, ptr addrspace(1) noundef align 4 %_arg_DoNotOptimize32) #0 !sycl_explicit_simd !3 {
entry:
  %0 = call spir_func i32 @_Z30__spirv_BuiltInSubgroupMaxSizev()
  %conv.i = zext i32 %0 to i64
  store i64 %conv.i, ptr addrspace(1) %_arg_DoNotOptimize, align 8
  %add.i = add i32 %0, 9
  store i32 %add.i, ptr addrspace(1) %_arg_DoNotOptimize32, align 4
  ret void
}
; CHECK: %conv.i = zext i32 1 to i64
; CHECK: store i64 %conv.i, ptr addrspace(1) %_arg_DoNotOptimize, align 8
; CHECK: %add.i = add i32 1, 9
; CHECK: store i32 %add.i, ptr addrspace(1) %_arg_DoNotOptimize32, align 4

declare spir_func i32 @_Z40__spirv_BuiltInSubgroupLocalInvocationIdv()
declare spir_func i32 @_Z27__spirv_BuiltInSubgroupSizev()
declare spir_func i32 @_Z30__spirv_BuiltInSubgroupMaxSizev()

attributes #0 = { "sycl-module-id"="a.cpp" }

!llvm.module.flags = !{!0}
!opencl.spir.version = !{!1}
!spirv.Source = !{!2}

!0 = !{i32 1, !"wchar_size", i32 4}
!1 = !{i32 1, i32 2}
!2 = !{i32 0, i32 100000}
!3 = !{}


