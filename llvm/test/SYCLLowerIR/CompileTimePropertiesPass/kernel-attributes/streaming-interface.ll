; Check conversion of sycl-streaming-interface attribute
; RUN: opt -passes="compile-time-properties" %s -S -o - | FileCheck %s --check-prefix CHECK-IR

; CHECK-IR-DAG: @pStreaming() #0 {{.*}}!ip_interface [[STREAMING:![0-9]+]] {
; Function Attrs: convergent norecurse
define weak_odr dso_local spir_kernel void @pStreaming() #0 {
entry:
  ret void
}

; CHECK-IR-DAG: @pStallFreeStreaming() #1 {{.*}}!ip_interface [[STALL_FREE_STREAMING:![0-9]+]] {
; Function Attrs: convergent norecurse
define weak_odr dso_local spir_kernel void @pStallFreeStreaming() #1 {
entry:
  ret void
}

attributes #0 = { convergent norecurse "frame-pointer"="all" "sycl-streaming-interface"="0" }
attributes #1 = { convergent norecurse "frame-pointer"="all" "sycl-streaming-interface"="1" }

!opencl.spir.version = !{!0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0}
!spirv.Source = !{!1, !1, !1, !1, !1, !1, !1, !1, !1, !1, !1}
!llvm.ident = !{!2, !2, !2, !2, !2, !2, !2, !2, !2, !2, !2}
!llvm.module.flags = !{!3, !4}

!0 = !{i32 1, i32 2}
!1 = !{i32 4, i32 100000}
!2 = !{!"clang version 13.0.0 (https://github.com/intel/llvm)"}
!3 = !{i32 1, !"wchar_size", i32 4}
!4 = !{i32 7, !"frame-pointer", i32 2}

; Confirm the decorations for the functions
; CHECK-IR-DAG: [[STREAMING]] = !{!"streaming"}
; CHECK-IR-DAG: [[STALL_FREE_STREAMING]] = !{!"streaming", !"stall_free_return"}
