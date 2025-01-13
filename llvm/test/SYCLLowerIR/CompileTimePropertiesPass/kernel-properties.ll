; RUN: opt -passes=compile-time-properties %s -S | FileCheck %s --check-prefix CHECK-IR

; CHECK-IR-DAG: @"_ZTSZZ4mainENK3$_0clERN2cl4sycl7handlerEE9TheKernel0"() #0 {{.*}}!intel_reqd_sub_group_size ![[SGSizeMD0:[0-9]+]] {{.*}}!reqd_work_group_size ![[WGSizeMD0:[0-9]+]]{{.*}}!work_group_size_hint ![[WGSizeHintMD0:[0-9]+]]
; Function Attrs: convergent norecurse
define weak_odr dso_local spir_kernel void @"_ZTSZZ4mainENK3$_0clERN2cl4sycl7handlerEE9TheKernel0"() #0 {
entry:
  ret void
}

; CHECK-IR-DAG: @"_ZTSZZ4mainENK3$_0clERN2cl4sycl7handlerEE9TheKernel1"() #1 {{.*}}!reqd_work_group_size ![[WGSizeMD1:[0-9]+]]{{.*}}!work_group_size_hint ![[WGSizeHintMD1:[0-9]+]]
; Function Attrs: convergent norecurse
define weak_odr dso_local spir_kernel void @"_ZTSZZ4mainENK3$_0clERN2cl4sycl7handlerEE9TheKernel1"() #1 {
entry:
  ret void
}

; CHECK-IR-DAG: @"_ZTSZZ4mainENK3$_0clERN2cl4sycl7handlerEE9TheKernel2"() #2 {{.*}}!reqd_work_group_size ![[WGSizeMD2:[0-9]+]]{{.*}}!work_group_size_hint ![[WGSizeHintMD2:[0-9]+]]
; Function Attrs: convergent norecurse
define weak_odr dso_local spir_kernel void @"_ZTSZZ4mainENK3$_0clERN2cl4sycl7handlerEE9TheKernel2"() #2 {
entry:
  ret void
}

attributes #0 = { convergent norecurse "frame-pointer"="all" "min-legal-vector-width"="0" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "sycl-module-id"="kernel_properties.cpp" "uniform-work-group-size"="true" "sycl-work-group-size"="1" "sycl-work-group-size-hint"="2" "sycl-sub-group-size"="3" }
attributes #1 = { convergent norecurse "frame-pointer"="all" "min-legal-vector-width"="0" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "sycl-module-id"="kernel_properties.cpp" "uniform-work-group-size"="true" "sycl-work-group-size"="4,5" "sycl-work-group-size-hint"="6,7" }
attributes #2 = { convergent norecurse "frame-pointer"="all" "min-legal-vector-width"="0" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "sycl-module-id"="kernel_properties.cpp" "uniform-work-group-size"="true" "sycl-work-group-size"="8,9,10" "sycl-work-group-size-hint"="11,12,13" }

!opencl.spir.version = !{!0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0}
!spirv.Source = !{!1, !1, !1, !1, !1, !1, !1, !1, !1, !1, !1}
!llvm.ident = !{!2, !2, !2, !2, !2, !2, !2, !2, !2, !2, !2}
!llvm.module.flags = !{!3, !4}

!0 = !{i32 1, i32 2}
!1 = !{i32 4, i32 100000}
!2 = !{!"clang version 13.0.0 (https://github.com/intel/llvm)"}
!3 = !{i32 1, !"wchar_size", i32 4}
!4 = !{i32 7, !"frame-pointer", i32 2}

; CHECK-IR-DAG: ![[SGSizeMD0]] = !{i32 3}
; CHECK-IR-DAG: ![[WGSizeMD0]] = !{[[SIZE_TY:i[0-9]+]] 1, [[SIZE_TY]] 1, [[SIZE_TY]] 1}
; CHECK-IR-DAG: ![[WGSizeHintMD0]] = !{[[SIZE_TY]] 2, [[SIZE_TY]] 1, [[SIZE_TY]] 1}
; CHECK-IR-DAG: ![[WGSizeMD1]] = !{[[SIZE_TY]] 5, [[SIZE_TY]] 4, [[SIZE_TY]] 1}
; CHECK-IR-DAG: ![[WGSizeHintMD1]] = !{[[SIZE_TY]] 7, [[SIZE_TY]] 6, [[SIZE_TY]] 1}
; CHECK-IR-DAG: ![[WGSizeMD2]] = !{[[SIZE_TY]] 10, [[SIZE_TY]] 9, [[SIZE_TY]] 8}
; CHECK-IR-DAG: ![[WGSizeHintMD2]] = !{[[SIZE_TY]] 13, [[SIZE_TY]] 12, [[SIZE_TY]] 11}
