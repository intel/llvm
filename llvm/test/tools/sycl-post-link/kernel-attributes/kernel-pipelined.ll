; Check conversion of sycl-pipelined attribute
; RUN: sycl-post-link --device-globals --ir-output-only %s -S -o - | FileCheck %s --check-prefix CHECK-IR

; CHECK-IR-DAG: @pipelineNegative() #0 {{.*}}!spirv.Decorations [[DEFAULT_PIPELINE:![0-9]+]] {
; Function Attrs: convergent norecurse
define weak_odr dso_local spir_kernel void @pipelineNegative() #0 {
entry:
  ret void
}

; CHECK-IR-DAG: @pipelineZero() #1 {{.*}}!spirv.Decorations [[NO_PIPELINE:![0-9]+]] {
; Function Attrs: convergent norecurse
define weak_odr dso_local spir_kernel void @pipelineZero() #1 {
entry:
  ret void
}

; CHECK-IR-DAG: @pipelinePositive() #2 {{.*}}!spirv.Decorations [[PIPELINE_WITH_II:![0-9]+]] {
; Function Attrs: convergent norecurse
define weak_odr dso_local spir_kernel void @pipelinePositive() #2 {
entry:
  ret void
}

attributes #0 = { convergent norecurse "frame-pointer"="all" "sycl-pipelined"="-1" }
attributes #1 = { convergent norecurse "frame-pointer"="all" "sycl-pipelined"="0" }
attributes #2 = { convergent norecurse "frame-pointer"="all" "sycl-pipelined"="2" }

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
; 5919 PipelineEnableINTEL
; 5917 InitiationIntervalINTEL
; CHECK-IR-DAG: [[DEFAULT_PIPELINE]] = !{[[PIPELINING_ON:![0-9]+]]}
; CHECK-IR-DAG: [[PIPELINING_ON]] = !{i32 5919, i32 1}
; CHECK-IR-DAG: [[NO_PIPELINE]] = !{[[PIPELINING_OFF:![0-9]+]]}
; CHECK-IR-DAG: [[PIPELINING_OFF]] = !{i32 5919, i32 0}
; CHECK-IR-DAG: [[PIPELINE_WITH_II]] = !{[[PIPELINING_ON]], [[II:![0-9]+]]}
; CHECK-IR-DAG: [[II]] = !{i32 5917, i32 2}
