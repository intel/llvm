; RUN: sycl-post-link -properties -split-esimd -split=source -S < %s -o %t.table
; RUN: FileCheck %s -input-file=%t.table
; RUN: FileCheck %s -input-file=%t_0.ll --check-prefixes CHECK-SYCL-IR-0
; RUN: FileCheck %s -input-file=%t_1.ll --check-prefixes CHECK-SYCL-IR-1
; RUN: FileCheck %s -input-file=%t_esimd_0.ll --check-prefixes CHECK-ESIMD-IR-0
; RUN: FileCheck %s -input-file=%t_esimd_1.ll --check-prefixes CHECK-ESIMD-IR-1

; This test checks that after we split SYCL and ESIMD kernels into
; separate modules, we split those two modules further according to
; -split option. In this case we have:
;   - 3 SYCL kernels: 2 in a.cpp, 1 in b.cpp
;   - 3 ESIMD kernels: 2 in a.cpp, 1 in b.cpp
; The module will be split into a total of 4 separate modules.

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

define dso_local spir_kernel void @ESIMD_kernel3() #1 !sycl_explicit_simd !3{
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

define dso_local spir_kernel void @SYCL_kernel3() #1 {
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
; CHECK-DAG: {{.*}}tmp_0.ll|{{.*}}_0.prop
; CHECK-DAG: {{.*}}tmp_1.ll|{{.*}}_1.prop
; CHECK-DAG: {{.*}}tmp_esimd_0.ll|{{.*}}_esimd_0.prop
; CHECK-DAG: {{.*}}tmp_esimd_1.ll|{{.*}}_esimd_1.prop

; CHECK-SYCL-IR-1-DAG: define dso_local spir_kernel void @SYCL_kernel1()
; CHECK-SYCL-IR-1-DAG: define dso_local spir_kernel void @SYCL_kernel2()
; CHECK-SYCL-IR-1-DAG: declare dso_local spir_func i64 @_Z28__spirv_GlobalInvocationId_xv()

; CHECK-SYCL-IR-0-DAG: define dso_local spir_kernel void @SYCL_kernel3()
; CHECK-SYCL-IR-0-DAG: declare dso_local spir_func i64 @_Z28__spirv_GlobalInvocationId_xv()

; CHECK-ESIMD-IR-1-DAG: define dso_local spir_kernel void @ESIMD_kernel1()
; CHECK-ESIMD-IR-1-DAG: define dso_local spir_kernel void @ESIMD_kernel2()
; CHECK-ESIMD-IR-1-DAG: declare dso_local spir_func i64 @_Z28__spirv_GlobalInvocationId_xv()

; CHECK-ESIMD-IR-0-DAG: define dso_local spir_kernel void @ESIMD_kernel3()
; CHECK-ESIMD-IR-0-DAG: declare dso_local spir_func i64 @_Z28__spirv_GlobalInvocationId_xv()
