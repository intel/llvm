; RUN: sycl-post-link --ir-output-only -split=auto -S %s -o %t.ll
; RUN: FileCheck %s -input-file=%t.ll

; This test checks that the --ir-output-only option writes a LLVM IR
; file instead of a table. In comparison with other tests, this one 
; checks that the option works OK with -split=auto.

target datalayout = "e-i64:64-v16:16-v24:32-v32:32-v48:64-v96:128-v192:256-v256:256-v512:512-v1024:1024"
target triple = "spir64-unknown-linux-sycldevice"

declare dso_local spir_func i64 @_Z28__spirv_GlobalInvocationId_xv()

define dso_local spir_kernel void @kernel1() #0 {
entry:
  %call = tail call spir_func i64 @_Z28__spirv_GlobalInvocationId_xv()
  ret void
}

define dso_local spir_kernel void @kernel2() #0 {
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

; CHECK: define dso_local spir_kernel void @kernel1()
; CHECK: define dso_local spir_kernel void @kernel2()
