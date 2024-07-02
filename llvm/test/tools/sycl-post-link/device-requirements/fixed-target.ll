; RUN: sycl-post-link -properties -split=auto < %s -o %t.files.table
; RUN: FileCheck %s -input-file=%t.files_0.prop

; CHECK: [SYCL/device requirements]
; CHECK: fixed_target=2|gAAAAAAAAAQAAAAA

source_filename = "main.cpp"
target datalayout = "e-i64:64-v16:16-v24:32-v32:32-v48:64-v96:128-v192:256-v256:256-v512:512-v1024:1024-n8:16:32:64"
target triple = "spir64-unknown-unknown"

$_ZTSZZ4mainENKUlRN4sycl3_V17handlerEE_clES2_EUlvE_ = comdat any

; Function Attrs: convergent mustprogress noinline norecurse optnone
define weak_odr dso_local spir_kernel void @_ZTSZZ4mainENKUlRN4sycl3_V17handlerEE_clES2_EUlvE_() #0 comdat {
entry:
  call spir_func void @_Z3foov()
  ret void
}

; Function Attrs: convergent mustprogress noinline norecurse nounwind optnone
define dso_local spir_func void @_Z3foov() !sycl_fixed_targets !3 {
entry:
  ret void
}

attributes #0 = { "sycl-module-id"="main.cpp" }

!sycl_aspects = !{!1, !2}

!1 = !{!"cpu", i32 1}
!2 = !{!"fp64", i32 6}
!3 = !{i32 1}
