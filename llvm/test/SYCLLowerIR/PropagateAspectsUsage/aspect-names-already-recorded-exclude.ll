; RUN: opt -passes=sycl-propagate-aspects-usage \
; RUN:   -sycl-propagate-aspects-usage-exclude-aspects=fp64 < %s -S \
; RUN:   | FileCheck %s --check-prefix=PROP
; RUN: opt -passes=sycl-propagate-aspects-usage,record-sycl-aspect-names \
; RUN:   -sycl-propagate-aspects-usage-exclude-aspects=fp64 < %s -S \
; RUN:   | FileCheck %s --check-prefix=PIPE
;
; Test checks the device pipeline when linked-in bitcode already carries
; !sycl_used_aspects in the {!"name", i32 value} form produced by
; RecordSYCLAspectNamesPass. This mimics clang, which runs aspect propagation
; with fp64 excluded at pipeline start and RecordSYCLAspectNamesPass at the end:
;  - lib_fp64 only uses the excluded aspect, so propagation keeps its metadata
;    as is (still in name/value form), and RecordSYCLAspectNamesPass must pass
;    the existing pairs through unchanged.
;  - lib_fp16_fp64 has a non-excluded aspect, so propagation rewrites its
;    metadata in plain integer form (keeping the excluded fp64), which
;    RecordSYCLAspectNamesPass then converts back to name/value pairs.

; PROP: define spir_kernel void @kernel_fp64() !sycl_fixed_targets ![[#]] {
; PIPE: define spir_kernel void @kernel_fp64() !sycl_fixed_targets ![[#]] {
define spir_kernel void @kernel_fp64() {
  call spir_func void @lib_fp64()
  ret void
}

; PROP: define spir_func void @lib_fp64() !sycl_used_aspects ![[#USED_FP64:]] {
; PIPE: define spir_func void @lib_fp64() !sycl_used_aspects ![[#USED_FP64:]] {
define spir_func void @lib_fp64() !sycl_used_aspects !10 {
  ret void
}

; PROP: define spir_kernel void @kernel_fp16() !sycl_used_aspects ![[#K_FP16:]]{{.*}} {
; PIPE: define spir_kernel void @kernel_fp16() !sycl_used_aspects ![[#K_FP16:]]{{.*}} {
define spir_kernel void @kernel_fp16() {
  call spir_func void @lib_fp16_fp64()
  ret void
}

; PROP: define spir_func void @lib_fp16_fp64() !sycl_used_aspects ![[#USED_BOTH:]] {
; PIPE: define spir_func void @lib_fp16_fp64() !sycl_used_aspects ![[#USED_BOTH:]] {
define spir_func void @lib_fp16_fp64() !sycl_used_aspects !11 {
  ret void
}

!sycl_aspects = !{!1, !2}
!1 = !{!"fp16", i32 5}
!2 = !{!"fp64", i32 6}

!10 = !{!2}
!11 = !{!1, !2}

; PROP-DAG: ![[#FP64:]] = !{!"fp64", i32 6}
; PROP-DAG: ![[#USED_FP64]] = !{![[#FP64]]}
; PROP-DAG: ![[#K_FP16]] = !{i32 5}
; PROP-DAG: ![[#USED_BOTH]] = !{i32 5, i32 6}

; PIPE-DAG: ![[#FP16:]] = !{!"fp16", i32 5}
; PIPE-DAG: ![[#FP64:]] = !{!"fp64", i32 6}
; PIPE-DAG: ![[#USED_FP64]] = !{![[#FP64]]}
; PIPE-DAG: ![[#K_FP16]] = !{![[#FP16]]}
; PIPE-DAG: ![[#USED_BOTH]] = !{![[#FP16]], ![[#FP64]]}
