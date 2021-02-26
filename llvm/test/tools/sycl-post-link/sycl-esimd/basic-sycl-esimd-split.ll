; RUN: sycl-post-link -split-esimd -S %s -o %t.table
; RUN: FileCheck %s -input-file=%t.table
; RUN: FileCheck %s -input-file=%t_0.ll --check-prefixes CHECK-SYCL-IR
; RUN: FileCheck %s -input-file=%t_esimd_0.ll --check-prefixes CHECK-ESIMD-IR
; RUN: FileCheck %s -input-file=%t_0.prop --check-prefixes CHECK-SYCL-PROP
; RUN: FileCheck %s -input-file=%t_esimd_0.prop --check-prefixes CHECK-ESIMD-PROP

; This is basic test of splitting SYCL and ESIMD kernels into separate modules.
; ESIMD module should have isEsimdImage=1 property set after splitting.

target datalayout = "e-i64:64-v16:16-v24:32-v32:32-v48:64-v96:128-v192:256-v256:256-v512:512-v1024:1024"
target triple = "spir64-unknown-linux-sycldevice"

declare dso_local spir_func i64 @_Z28__spirv_GlobalInvocationId_xv()

define dso_local spir_kernel void @ESIMD_kernel() #0 !sycl_explicit_simd !3{
entry:
  %call = tail call spir_func i64 @_Z28__spirv_GlobalInvocationId_xv()
  ret void
}

define dso_local spir_kernel void @SYCL_kernel() #0 {
entry:
  %call = tail call spir_func i64 @_Z28__spirv_GlobalInvocationId_xv()
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

; CHECK: [Code|Properties]
; CHECK: {{.*}}_0.ll|{{.*}}_0.prop
; CHECK: {{.*}}_esimd_0.ll|{{.*}}_esimd_0.prop

; CHECK-SYCL-IR-DAG: define dso_local spir_kernel void @SYCL_kernel()
; CHECK-SYCL-IR-DAG: declare dso_local spir_func i64 @_Z28__spirv_GlobalInvocationId_xv()

; CHECK-SYCL-PROP-NOT: isEsimdImage=1|1

; CHECK-ESIMD-IR-DAG: define dso_local spir_kernel void @ESIMD_kernel()
; CHECK-ESIMD-IR-DAG: declare dso_local spir_func i64 @_Z28__spirv_GlobalInvocationId_xv()

; CHECK-ESIMD-PROP: [SYCL/misc properties]
; CHECK-ESIMD-PROP: isEsimdImage=1|1
