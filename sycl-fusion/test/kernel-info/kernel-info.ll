; RUN: opt -load-pass-plugin %shlibdir/SYCLKernelFusion%shlibext\
; RUN: -passes=print-sycl-module-info -disable-output %s | FileCheck %s

target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"
target triple = "spir64-unknown-unknown"

!0 = !{
  !"Accessor", !"StdLayout", !"StdLayout", !"StdLayout", !"Accessor",
  !"Sampler", !"Pointer", !"SpecConstantBuffer", !"Stream",
  !"StdLayout", !"Invalid", !"FooBar"}
!1 = !{i8 1, i8 0, i8 0, i8 1, i8 1, i8 0, i8 0, i8 1, i8 1, i8 0, i8 0, i8 1}
!2 = !{!"_ZTSZZ4mainENKUlRN4sycl3_V17handlerEE_clES2_E9KernelOne", !0, !1}
!3 = !{!"_ZTSZZ4mainENKUlRN4sycl3_V17handlerEE0_clES2_E9KernelTwo", !0, !1,
       !{!"work_group_size_hint", i32 1, i32 2, i32 3},
       !{!"reqd_work_group_size", i32 4, i32 5, i32 6}}
!sycl.moduleinfo = !{!2, !3}

; Test scenario: Test if the analysis and the printing pass work correctly
; by checking that loading the info from the file in the analysis and 
; subsequently printing the module info matches the file content.

; CHECK-LABEL: KernelName: _ZTSZZ4mainENKUlRN4sycl3_V17handlerEE_clES2_E9KernelOne
; CHECK-NEXT:    Args:
; CHECK-NEXT:      Kinds:  Accessor, StdLayout, StdLayout, StdLayout, Accessor,
; CHECK-SAME:              Sampler, Pointer, SpecConstantBuffer, Stream,
; CHECK-SAME:              StdLayout, Invalid, Invalid
; CHECK-NEXT:      Mask:   1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1
; CHECK-LABEL: KernelName: _ZTSZZ4mainENKUlRN4sycl3_V17handlerEE0_clES2_E9KernelTwo
; CHECK-NEXT:    Args:
; CHECK-NEXT:      Kinds:  Accessor, StdLayout, StdLayout, StdLayout, Accessor,
; CHECK-SAME:              Sampler, Pointer, SpecConstantBuffer, Stream,
; CHECK-SAME:              StdLayout, Invalid, Invalid
; CHECK-NEXT:      Mask:   1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1
; CHECK-NEXT:    Attributes:
; CHECK-NEXT:        work_group_size_hint: 1, 2, 3
; CHECK-NEXT:        reqd_work_group_size: 4, 5, 6
