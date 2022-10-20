; RUN: opt -passes=sycl-propagate-aspects-usage -sycl-propagate-aspects-usage-fixed-targets=host,cpu,gpu %s -S | FileCheck %s

source_filename = "main.cpp"
target datalayout = "e-i64:64-v16:16-v24:32-v32:32-v48:64-v96:128-v192:256-v256:256-v512:512-v1024:1024-n8:16:32:64"
target triple = "spir64-unknown-unknown"

; CHECK: void @kernel(){{.*}}!sycl_fixed_targets ![[#MDNUM:]]
define weak_odr dso_local spir_kernel void @kernel() {
entry:
  ret void
}

!sycl_aspects = !{!0, !1, !2, !3}

; CHECK: ![[#MDNUM]] = !{i32 0, i32 1, i32 2}

!0 = !{!"host", i32 0}
!1 = !{!"cpu", i32 1}
!2 = !{!"gpu", i32 2}
!3 = !{!"fp64", i32 6}
