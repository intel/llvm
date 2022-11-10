; RUN: sycl-post-link -split=auto %s -o %t.files.table
; RUN: FileCheck %s -input-file=%t.files_0.prop --check-prefix CHECK-AUTO

; RUN: sycl-post-link -split=kernel %s -o %t.files.table
; RUN: FileCheck %s -input-file=%t.files_0.prop --check-prefix CHECK-KERNEL-0
; RUN: FileCheck %s -input-file=%t.files_1.prop --check-prefix CHECK-KERNEL-1
; RUN: FileCheck %s -input-file=%t.files_2.prop --check-prefix CHECK-KERNEL-2

; CHECK: aspects=2|gBAAAAAAAAQBAAAAGAAAAkAAAAA
; CHECK-KERNEL-0: aspects=2|gBAAAAAAAAQBAAAAGAAAAkAAAAA
; CHECK-KERNEL-1: aspects=2|ABAAAAAAAAQBAAAAGAAAAA
; CHECK-KERNEL-2: aspects=2|gAAAAAAAAAQBAAAA

source_filename = "source.cpp"
target datalayout = "e-i64:64-v16:16-v24:32-v32:32-v48:64-v96:128-v192:256-v256:256-v512:512-v1024:1024-n8:16:32:64"
target triple = "spir64-unknown-unknown"

$_ZTSZZ4mainENKUlRN4sycl3_V17handlerEE_clES2_EUlvE_ = comdat any

$_ZTSZZ4mainENKUlRN4sycl3_V17handlerEE0_clES2_EUlvE_ = comdat any

$_ZTSZZ4mainENKUlRN4sycl3_V17handlerEE1_clES2_EUlvE_ = comdat any

define weak_odr dso_local spir_kernel void @_ZTSZZ4mainENKUlRN4sycl3_V17handlerEE_clES2_EUlvE_() #0 {
entry:
  call spir_func void @_Z3foov()
  ret void
}

define dso_local spir_func void @_Z3foov() !sycl_declared_aspects !47 {
entry:
  ret void
}

define weak_odr dso_local spir_kernel void @_ZTSZZ4mainENKUlRN4sycl3_V17handlerEE0_clES2_EUlvE_() #0 {
entry:
  call spir_func void @_Z3barv()
  ret void
}

define dso_local spir_func void @_Z3barv() !sycl_declared_aspects !50 {
entry:
  ret void
}

define weak_odr dso_local spir_kernel void @_ZTSZZ4mainENKUlRN4sycl3_V17handlerEE1_clES2_EUlvE_() #0 {
entry:
  call spir_func void @_Z3bazv()
  ret void
}

define dso_local spir_func void @_Z3bazv() !sycl_declared_aspects !53 {
entry:
  ret void
}

attributes #0 = { "sycl-module-id"="throw-exception-for-unsupported-aspect.cpp" }


!sycl_aspects = !{!9, !10, !11}

!9 = !{!"fp16", i32 5}
!10 = !{!"fp64", i32 6}
!11 = !{!"image", i32 9}
!47 = !{i32 5}
!50 = !{i32 5, i32 6}
!53 = !{i32 5, i32 6, i32 9}
