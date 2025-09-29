; This test checks that the sycl-post-link ouputs registered kernel data 
; from !sycl_registered_kernels metadata into the SYCL/registerd_kernels section.

; RUN: sycl-post-link %s -properties -split=auto -o %t.table
; RUN: FileCheck %s -input-file=%t_0.prop --check-prefixes=CHECK-WITH-ASPECT,CHECK \
; RUN:   --implicit-check-not=kernel_with_aspects
; RUN: FileCheck %s -input-file=%t_1.prop --check-prefixes=CHECK-NO-ASPECT,CHECK

!sycl_registered_kernels = !{!4}
!4 = !{!5, !6, !7, !8, !9, !10, !11, !12, !13, !14, !16, !17, !18, !19}

; Both splits should contain the registered kernel data.
; CHECK: [SYCL/registered kernels]

; For each entry in !sycl_registered_kernels, an entry
; mapping the registered name to the mangled name is added in the
; [SYCL/registered kernels] if it references a kernel that appears
; in the split. (Although in the prop files, the
; mapped values are base64 encoded, so just using simplifed check
; with a regex.)
; CHECK-NO-ASPECT-NEXT: foo=2|{{[A-Za-z0-9+/]+}}
!5 = !{!"foo", !"_Z17__sycl_kernel_foov"}
define spir_kernel void @_Z17__sycl_kernel_foov() {
    ret void
}

; CHECK-NO-ASPECT-NEXT: foo3=2|{{[A-Za-z0-9+/]+}}
!6 = !{!"foo3", !"_Z18__sycl_kernel_ff_4v"}
define spir_kernel void @_Z18__sycl_kernel_ff_4v() {
    ret void
}

; CHECK-NO-ASPECT-NEXT: iota=2|{{[A-Za-z0-9+/]+}}
!7 = !{!"iota", !"_Z18__sycl_kernel_iotaiPi"}
define spir_kernel void @_Z18__sycl_kernel_iotaiPi() {
    ret void
}

; CHECK-NO-ASPECT-NEXT: inst temp=2|{{[A-Za-z0-9+/]+}}
!8 = !{!"inst temp", !"_Z22__sycl_kernel_tempfoo2IiEvT_"}
define spir_kernel void @_Z22__sycl_kernel_tempfoo2IiEvT_() {
    ret void
}

; CHECK-NO-ASPECT-NEXT: def spec=2|{{[A-Za-z0-9+/]+}}
!9 = !{!"def spec", !"_Z22__sycl_kernel_tempfoo2IsEvT_"}
define spir_kernel void @_Z22__sycl_kernel_tempfoo2IsEvT_() {
    ret void
}

; CHECK-NO-ASPECT-NEXT: decl temp=2|{{[A-Za-z0-9+/]+}}
!10 = !{!"decl temp", !"_Z21__sycl_kernel_tempfooIiEvT_"}
define spir_kernel void @_Z21__sycl_kernel_tempfooIiEvT_() {
    ret void
}

; CHECK-NO-ASPECT-NEXT: decl spec=2|{{[A-Za-z0-9+/]+}}
!11 = !{!"decl spec", !"_Z22__sycl_kernel_tempfoo2IfEvT_"}
define spir_kernel void @_Z22__sycl_kernel_tempfoo2IfEvT_() {
    ret void
}

; CHECK-NO-ASPECT-NEXT: nontype=2|{{[A-Za-z0-9+/]+}}
!12 = !{!"nontype", !"_Z22__sycl_kernel_tempfoo3ILi5EEvv"}
define spir_kernel void @_Z22__sycl_kernel_tempfoo3ILi5EEvv() {
    ret void
}

; CHECK-NO-ASPECT-NEXT: non-temp=2|{{[A-Za-z0-9+/]+}}
!13 = !{!"decl non-temp", !"_Z17__sycl_kernel_barv"}
define spir_kernel void @_Z17__sycl_kernel_barv() {
    ret void
}

!14 = !{!"kernel_with_aspects", !"kernel_with_aspects"}
!15 = !{i32 1}
; CHECK-WITH-ASPECT-NEXT: kernel_with_aspects=2|{{[A-Za-z0-9+/]+}}
define spir_kernel void @kernel_with_aspects() !sycl_used_aspects !15 {
    ret void
}

; Data with incorrect format should be ignored.
; CHECK-NOT: incorrect_data_format
!16 = !{!"incorrect_data_format"}

!17 = !{!"bar", !"_Z3barv"}
!18 = !{!"bar", !"_Z3barv"}
!19 = !{!"(void(*)())bar", !"_Z3barv"}
; CHECK-NO-ASPECT-NEXT: bar=2|[[BAR:[A-Za-z0-9+/]+]]
; CHECK-NO-ASPECT-NEXT: (void(*)())bar=2|[[BAR]]
define spir_kernel void @_Z3barv() {
    ret void
}
