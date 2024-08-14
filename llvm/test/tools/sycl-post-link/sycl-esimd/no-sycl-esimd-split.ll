; RUN: sycl-post-link -properties -split=source -S < %s -o %t.table
; RUN: FileCheck %s -input-file=%t.table
; RUN: FileCheck %s -input-file=%t_0.ll --check-prefixes CHECK-IR-0
; RUN: FileCheck %s -input-file=%t_1.ll --check-prefixes CHECK-IR-1

; This test checks that if no '-split-esimd' provided, ther is no 
; splitting of SYCL and ESIMD kernels into separate modules.
; However, the rest of the splitting still happens according to
; the '-split=' option.

target datalayout = "e-i64:64-v16:16-v24:32-v32:32-v48:64-v96:128-v192:256-v256:256-v512:512-v1024:1024"
target triple = "spir64-unknown-linux"

declare dso_local spir_func i64 @_Z28__spirv_GlobalInvocationId_xv()

define dso_local spir_kernel void @ESIMD_kernel() #0 !sycl_explicit_simd !3{
entry:
  %call = tail call spir_func i64 @_Z28__spirv_GlobalInvocationId_xv()
  ret void
}

define dso_local spir_kernel void @SYCL_kernel1() #0 {
entry:
  %call = tail call spir_func i64 @_Z28__spirv_GlobalInvocationId_xv()
  ret void
}

define dso_local spir_kernel void @SYCL_kernel2() #1 {
entry:
  %call = tail call spir_func i64 @_Z28__spirv_GlobalInvocationId_xv()
  ret void
}

attributes #0 = { "sycl-module-id"="a.cpp" }
attributes #1 = { "sycl-module-id"="b.cpp" }

!llvm.module.flags = !{!0}
!opencl.spir.version = !{!1}
!spirv.Source = !{!2}

!0 = !{i32 1, !"wchar_size", i32 4}
!1 = !{i32 1, i32 2}
!2 = !{i32 0, i32 100000}
!3 = !{}

; CHECK: [Code|Properties]
; CHECK: {{.*}}_0.ll|{{.*}}_0.prop
; CHECK: {{.*}}_1.ll|{{.*}}_1.prop

; CHECK-IR-1-DAG: define dso_local spir_kernel void @SYCL_kernel1()
; CHECK-IR-1-DAG: define dso_local spir_kernel void @ESIMD_kernel()
; CHECK-IR-1-DAG: declare dso_local spir_func i64 @_Z28__spirv_GlobalInvocationId_xv()

; CHECK-IR-0-DAG: define dso_local spir_kernel void @SYCL_kernel2()
; CHECK-IR-0-DAG: declare dso_local spir_func i64 @_Z28__spirv_GlobalInvocationId_xv()
