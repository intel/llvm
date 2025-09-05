; RUN: sycl-post-link -properties -split-esimd -symbols -S < %s -o %t.table
; RUN: FileCheck %s -input-file=%t.table
; RUN: FileCheck %s -input-file=%t_0.sym --check-prefixes CHECK-SYCL-SYM
; RUN: FileCheck %s -input-file=%t_esimd_0.sym --check-prefixes CHECK-ESIMD-SYM

; This test checks symbols generation when we split SYCL and ESIMD kernels into
; separate modules.

target datalayout = "e-i64:64-v16:16-v24:32-v32:32-v48:64-v96:128-v192:256-v256:256-v512:512-v1024:1024"
target triple = "spir64-unknown-linux"

declare dso_local spir_func i64 @_Z28__spirv_GlobalInvocationId_xv()

define dso_local spir_kernel void @ESIMD_kernel1() #0 !sycl_explicit_simd !3{
entry:
  %call = tail call spir_func i64 @_Z28__spirv_GlobalInvocationId_xv()
  ret void
}

define dso_local spir_kernel void @ESIMD_kernel2() #0 !sycl_explicit_simd !3{
entry:
  %call = tail call spir_func i64 @_Z28__spirv_GlobalInvocationId_xv()
  ret void
}

define dso_local spir_kernel void @SYCL_kernel1() #0 {
entry:
  %call = tail call spir_func i64 @_Z28__spirv_GlobalInvocationId_xv()
  ret void
}

define dso_local spir_kernel void @SYCL_kernel2() #0 {
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

; CHECK: [Code|Properties|Symbols]
; CHECK-DAG: {{.*}}tmp_0.ll|{{.*}}_0.prop|{{.*}}_0.sym
; CHECK-DAG: {{.*}}tmp_esimd_0.ll|{{.*}}_esimd_0.prop|{{.*}}_esimd_0.sym

; CHECK-SYCL-SYM: SYCL_kernel1
; CHECK-SYCL-SYM: SYCL_kernel2

; CHECK-ESIMD-SYM: ESIMD_kernel1
; CHECK-ESIMD-SYM: ESIMD_kernel2
