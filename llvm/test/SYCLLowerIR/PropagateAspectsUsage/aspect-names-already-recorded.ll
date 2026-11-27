; RUN: opt -passes=sycl-propagate-aspects-usage < %s -S | FileCheck %s
; RUN: opt -passes=sycl-propagate-aspects-usage < %s -disable-output 2>&1 \
; RUN:   | FileCheck %s --check-prefix=WARN
;
; Test checks that the pass accepts !sycl_used_aspects and
; !sycl_declared_aspects metadata that has already been rewritten by
; RecordSYCLAspectNamesPass, i.e. operands of the form {!"name", i32 value}
; instead of plain i32 values. Such metadata appears when device bitcode that
; has already gone through the full device pipeline is linked into the module
; (e.g. via -mlink-builtin-bitcode) before aspect propagation runs.
;
; K1 -> lib_used              ; used:     {{"fp16", 5}}
; K2 -> lib_mixed             ; used:     {{"fp16", 5}, 2}
; K3 -> lib_declared_match    ; declared: {{"fp16", 5}}, uses fp16 type
; K4 -> lib_declared_mismatch ; declared: {{"gpu", 2}},  uses fp16 type

%Optional.fp16 = type { i32 }

; CHECK: define spir_kernel void @kernel1() !sycl_used_aspects ![[#ID1:]]{{.*}} {
define spir_kernel void @kernel1() {
  call spir_func void @lib_used()
  ret void
}

; Existing name/value pairs are read and merged into the new metadata.
; CHECK: define spir_func void @lib_used() !sycl_used_aspects ![[#ID1]] {
define spir_func void @lib_used() !sycl_used_aspects !10 {
  ret void
}

; CHECK: define spir_kernel void @kernel2() !sycl_used_aspects ![[#ID2:]]{{.*}} {
define spir_kernel void @kernel2() {
  call spir_func void @lib_mixed()
  ret void
}

; A mix of name/value pairs and plain values (RecordSYCLAspectNamesPass leaves
; values without an entry in !sycl_aspects as plain integers).
; CHECK: define spir_func void @lib_mixed() !sycl_used_aspects ![[#ID2]] {
define spir_func void @lib_mixed() !sycl_used_aspects !11 {
  ret void
}

; CHECK: define spir_kernel void @kernel3() !sycl_used_aspects ![[#ID1]]{{.*}} {
define spir_kernel void @kernel3() {
  call spir_func void @lib_declared_match()
  ret void
}

; Declared aspects in name/value form are read correctly: fp16 is declared,
; so no warning is emitted.
; CHECK: define spir_func void @lib_declared_match() !sycl_used_aspects ![[#ID1]] !sycl_declared_aspects ![[#DECL_FP16:]] {
define spir_func void @lib_declared_match() !sycl_declared_aspects !10 {
  %tmp = alloca %Optional.fp16
  ret void
}

; CHECK: define spir_kernel void @kernel4() !sycl_used_aspects ![[#ID2]]{{.*}} {
define spir_kernel void @kernel4() {
  call spir_func void @lib_declared_mismatch()
  ret void
}

; Only gpu is declared, but fp16 is used, so a warning is emitted.
; CHECK: define spir_func void @lib_declared_mismatch() !sycl_used_aspects ![[#ID2]] !sycl_declared_aspects ![[#DECL_GPU:]] {
define spir_func void @lib_declared_mismatch() !sycl_declared_aspects !12 {
  %tmp = alloca %Optional.fp16
  ret void
}

; WARN-NOT: lib_declared_match uses aspect
; WARN: lib_declared_mismatch uses aspect "fp16" but does not specify that aspect as available in its "sycl::device_has" attribute
; WARN-NOT: lib_declared_match uses aspect

!sycl_types_that_use_aspects = !{!0}
!0 = !{!"Optional.fp16", i32 5}

!sycl_aspects = !{!1, !2, !3}
!1 = !{!"gpu", i32 2}
!2 = !{!"fp16", i32 5}
!3 = !{!"fp64", i32 6}

!10 = !{!2}
!11 = !{!2, i32 2}
!12 = !{!1}

; Declared aspects metadata is left untouched.
; CHECK-DAG: ![[#FP16:]] = !{!"fp16", i32 5}
; CHECK-DAG: ![[#GPU:]] = !{!"gpu", i32 2}
; CHECK-DAG: ![[#DECL_FP16]] = !{![[#FP16]]}
; CHECK-DAG: ![[#DECL_GPU]] = !{![[#GPU]]}
; Used aspects metadata is rewritten in the plain integer form.
; CHECK-DAG: ![[#ID1]] = !{i32 5}
; CHECK-DAG: ![[#ID2]] = !{i32 5, i32 2}
