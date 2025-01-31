; This test checks that the sycl-post-link ouputs registered kernel data 
; from !sycl_registered_kernels metadata into the SYCL/registerd_kernels section.
; RUN: sycl-post-link %s -properties -o %t.table
; RUN: FileCheck %s -input-file=%t_0.prop

!sycl_registered_kernels = !{!4}
!4 = !{!5, !6, !7, !8, !9, !10, !11, !12, !13}

; CHECK: [SYCL/registered kernels]

; For each entry in !sycl_registered_kernels, an entry
; mapping the registered name to the mangled name is added in the
; SYCL/registered kernels. (Although in the prop files, the
; mapped values are base64 encoded, so just using simplifed check
; with a regex.)
; CHECK-NEXT: foo=2|{{[A-Za-z0-9+/]+}}
!5 = !{!"foo", !"_Z17__sycl_kernel_foov"}

; CHECK-NEXT: foo3=2|{{[A-Za-z0-9+/]+}}
!6 = !{!"foo3", !"_Z18__sycl_kernel_ff_4v"}

; CHECK-NEXT: iota=2|{{[A-Za-z0-9+/]+}}
!7 = !{!"iota", !"_Z18__sycl_kernel_iotaiPi"}

; CHECK-NEXT: inst temp=2|{{[A-Za-z0-9+/]+}}
!8 = !{!"inst temp", !"_Z22__sycl_kernel_tempfoo2IiEvT_"}

; CHECK-NEXT: def spec=2|{{[A-Za-z0-9+/]+}}
!9 = !{!"def spec", !"_Z22__sycl_kernel_tempfoo2IsEvT_"}

; CHECK-NEXT: decl temp=2|{{[A-Za-z0-9+/]+}}
!10 = !{!"decl temp", !"_Z21__sycl_kernel_tempfooIiEvT_"}

; CHECK-NEXT: decl spec=2|{{[A-Za-z0-9+/]+}}
!11 = !{!"decl spec", !"_Z22__sycl_kernel_tempfoo2IfEvT_"}

; CHECK-NEXT: nontype=2|{{[A-Za-z0-9+/]+}}
!12 = !{!"nontype", !"_Z22__sycl_kernel_tempfoo3ILi5EEvv"}

; CHECK-NEXT: non-temp=2|{{[A-Za-z0-9+/]+}}
!13 = !{!"decl non-temp", !"_Z17__sycl_kernel_barv"}
