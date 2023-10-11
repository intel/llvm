; RUN: opt -passes=compile-time-properties %s -S | FileCheck %s


define spir_kernel void @"Kernel0"() #0 {
entry:
  ret void
}

define spir_kernel void @"Kernel1"() #1 {
entry:
  ret void
}

define spir_kernel void @"Kernel2"() #2 {
entry:
  ret void
}

define spir_kernel void @"Kernel3"() #3 {
entry:
  ret void
}

define spir_kernel void @"Kernel4"() #4 {
entry:
  ret void
}

define spir_kernel void @"Kernel5"() #5 {
entry:
  ret void
}

define spir_kernel void @"Kernel6"() #6 {
entry:
  ret void
}

define spir_kernel void @"Kernel7"() #7 {
entry:
  ret void
}

define spir_kernel void @"Kernel8"() #8 {
entry:
  ret void
}

define spir_kernel void @"Kernel9"() #9 {
entry:
  ret void
}

define spir_kernel void @"Kernel10"() #10 {
entry:
  ret void
}

define spir_kernel void @"Kernel11"() #11 {
entry:
  ret void
}

define spir_kernel void @"Kernel12"() #12 {
entry:
  ret void
}

; SPIRV execution modes for FP control.   | BitMask
; ROUNDING_MODE_RTE = 4462;               | 00000000000
; ROUNDING_MODE_RTP_INTEL = 5620;         | 00000010000
; ROUNDING_MODE_RTN_INTEL = 5621;         | 00000100000
; ROUNDING_MODE_RTZ = 4463;               | 00000110000
; DEMORM_FLUSH_TO_ZERO = 4460;            | 00000000000
; DENORM_PRESERVE (double) = 4459;        | 00001000000
; DENORM_PRESERVE (float) = 4459;         | 00010000000
; DENORM_PRESERVE (half) = 4459;          | 10000000000
; FLOATING_POINT_MODE_ALT_INTEL = 5622;   | 00000000001
; FLOATING_POINT_MODE_IEEE_INTEL = 5623;  | 00000000000

; rte + ftz + ieee (Default)
; CHECK: !0 = !{ptr @Kernel0, i32 4462, i32 64}
; CHECK: !1 = !{ptr @Kernel0, i32 4462, i32 32}
; CHECK: !2 = !{ptr @Kernel0, i32 4462, i32 16}
; CHECK: !3 = !{ptr @Kernel0, i32 5623, i32 64}
; CHECK: !4 = !{ptr @Kernel0, i32 5623, i32 32}
; CHECK: !5 = !{ptr @Kernel0, i32 5623, i32 16}
; CHECK: !6 = !{ptr @Kernel0, i32 4460, i32 64}
; CHECK: !7 = !{ptr @Kernel0, i32 4460, i32 32}
; CHECK: !8 = !{ptr @Kernel0, i32 4460, i32 16}
attributes #0 = { "sycl-floating-point-control"="0" }

; rtp + ftz + ieee
; CHECK: !9 = !{ptr @Kernel1, i32 5620, i32 64}
; CHECK: !10 = !{ptr @Kernel1, i32 5620, i32 32}
; CHECK: !11 = !{ptr @Kernel1, i32 5620, i32 16}
; CHECK: !12 = !{ptr @Kernel1, i32 5623, i32 64}
; CHECK: !13 = !{ptr @Kernel1, i32 5623, i32 32}
; CHECK: !14 = !{ptr @Kernel1, i32 5623, i32 16}
; CHECK: !15 = !{ptr @Kernel1, i32 4460, i32 64}
; CHECK: !16 = !{ptr @Kernel1, i32 4460, i32 32}
; CHECK: !17 = !{ptr @Kernel1, i32 4460, i32 16}
attributes #1 = { "sycl-floating-point-control"="16" }

; rtn + ftz + ieee
; CHECK: !18 = !{ptr @Kernel2, i32 5621, i32 64}
; CHECK: !19 = !{ptr @Kernel2, i32 5621, i32 32}
; CHECK: !20 = !{ptr @Kernel2, i32 5621, i32 16}
; CHECK: !21 = !{ptr @Kernel2, i32 5623, i32 64}
; CHECK: !22 = !{ptr @Kernel2, i32 5623, i32 32}
; CHECK: !23 = !{ptr @Kernel2, i32 5623, i32 16}
; CHECK: !24 = !{ptr @Kernel2, i32 4460, i32 64}
; CHECK: !25 = !{ptr @Kernel2, i32 4460, i32 32}
; CHECK: !26 = !{ptr @Kernel2, i32 4460, i32 16}
attributes #2 = { "sycl-floating-point-control"="32" }

; rtz + ftz + ieee
; CHECK: !27 = !{ptr @Kernel3, i32 4463, i32 64}
; CHECK: !28 = !{ptr @Kernel3, i32 4463, i32 32}
; CHECK: !29 = !{ptr @Kernel3, i32 4463, i32 16}
; CHECK: !30 = !{ptr @Kernel3, i32 5623, i32 64}
; CHECK: !31 = !{ptr @Kernel3, i32 5623, i32 32}
; CHECK: !32 = !{ptr @Kernel3, i32 5623, i32 16}
; CHECK: !33 = !{ptr @Kernel3, i32 4460, i32 64}
; CHECK: !34 = !{ptr @Kernel3, i32 4460, i32 32}
; CHECK: !35 = !{ptr @Kernel3, i32 4460, i32 16}
attributes #3 = { "sycl-floating-point-control"="48" }

; rte + denorm_preserve(double) + ieee
; CHECK: !36 = !{ptr @Kernel4, i32 4462, i32 64}
; CHECK: !37 = !{ptr @Kernel4, i32 4462, i32 32}
; CHECK: !38 = !{ptr @Kernel4, i32 4462, i32 16}
; CHECK: !39 = !{ptr @Kernel4, i32 5623, i32 64}
; CHECK: !40 = !{ptr @Kernel4, i32 5623, i32 32}
; CHECK: !41 = !{ptr @Kernel4, i32 5623, i32 16}
; CHECK: !42 = !{ptr @Kernel4, i32 4459, i32 64}
attributes #4 = { "sycl-floating-point-control"="64" }

; rte + denorm_preserve(float) + ieee
; CHECK: !43 = !{ptr @Kernel5, i32 4462, i32 64}
; CHECK: !44 = !{ptr @Kernel5, i32 4462, i32 32}
; CHECK: !45 = !{ptr @Kernel5, i32 4462, i32 16}
; CHECK: !46 = !{ptr @Kernel5, i32 5623, i32 64}
; CHECK: !47 = !{ptr @Kernel5, i32 5623, i32 32}
; CHECK: !48 = !{ptr @Kernel5, i32 5623, i32 16}
; CHECK: !49 = !{ptr @Kernel5, i32 4459, i32 32}
attributes #5 = { "sycl-floating-point-control"="128" }

; rte + denorm_preserve(half) + ieee
; CHECK: !50 = !{ptr @Kernel6, i32 4462, i32 64}
; CHECK: !51 = !{ptr @Kernel6, i32 4462, i32 32}
; CHECK: !52 = !{ptr @Kernel6, i32 4462, i32 16}
; CHECK: !53 = !{ptr @Kernel6, i32 5623, i32 64}
; CHECK: !54 = !{ptr @Kernel6, i32 5623, i32 32}
; CHECK: !55 = !{ptr @Kernel6, i32 5623, i32 16}
; CHECK: !56 = !{ptr @Kernel6, i32 4459, i32 16}
attributes #6 = { "sycl-floating-point-control"="1024" }

; rte + denorm_allow + ieee
; CHECK: !57 = !{ptr @Kernel7, i32 4462, i32 64}
; CHECK: !58 = !{ptr @Kernel7, i32 4462, i32 32}
; CHECK: !59 = !{ptr @Kernel7, i32 4462, i32 16}
; CHECK: !60 = !{ptr @Kernel7, i32 5623, i32 64}
; CHECK: !61 = !{ptr @Kernel7, i32 5623, i32 32}
; CHECK: !62 = !{ptr @Kernel7, i32 5623, i32 16}
; CHECK: !63 = !{ptr @Kernel7, i32 4459, i32 16}
; CHECK: !64 = !{ptr @Kernel7, i32 4459, i32 32}
; CHECK: !65 = !{ptr @Kernel7, i32 4459, i32 64}
attributes #7 = { "sycl-floating-point-control"="1216" }

; rte + ftz + alt
; CHECK: !66 = !{ptr @Kernel8, i32 4462, i32 64}
; CHECK: !67 = !{ptr @Kernel8, i32 4462, i32 32}
; CHECK: !68 = !{ptr @Kernel8, i32 4462, i32 16}
; CHECK: !69 = !{ptr @Kernel8, i32 5622, i32 64}
; CHECK: !70 = !{ptr @Kernel8, i32 5622, i32 32}
; CHECK: !71 = !{ptr @Kernel8, i32 5622, i32 16}
; CHECK: !72 = !{ptr @Kernel8, i32 4460, i32 64}
; CHECK: !73 = !{ptr @Kernel8, i32 4460, i32 32}
; CHECK: !74 = !{ptr @Kernel8, i32 4460, i32 16}
attributes #8 = { "sycl-floating-point-control"="1" }

; rtz + denorm_preserve(double) + ieee
; CHECK: !75 = !{ptr @Kernel9, i32 4463, i32 64}
; CHECK: !76 = !{ptr @Kernel9, i32 4463, i32 32}
; CHECK: !77 = !{ptr @Kernel9, i32 4463, i32 16}
; CHECK: !78 = !{ptr @Kernel9, i32 5623, i32 64}
; CHECK: !79 = !{ptr @Kernel9, i32 5623, i32 32}
; CHECK: !80 = !{ptr @Kernel9, i32 5623, i32 16}
; CHECK: !81 = !{ptr @Kernel9, i32 4459, i32 64}
attributes #9 = { "sycl-floating-point-control"="112" }

; rtp + denorm_preserve(float) + ieee
; CHECK: !82 = !{ptr @Kernel10, i32 5620, i32 64}
; CHECK: !83 = !{ptr @Kernel10, i32 5620, i32 32}
; CHECK: !84 = !{ptr @Kernel10, i32 5620, i32 16}
; CHECK: !85 = !{ptr @Kernel10, i32 5623, i32 64}
; CHECK: !86 = !{ptr @Kernel10, i32 5623, i32 32}
; CHECK: !87 = !{ptr @Kernel10, i32 5623, i32 16}
; CHECK: !88 = !{ptr @Kernel10, i32 4459, i32 32}
attributes #10 = { "sycl-floating-point-control"="144" }

; rtp + denorm_preserve(float) + alt
; CHECK: !89 = !{ptr @Kernel11, i32 5620, i32 64}
; CHECK: !90 = !{ptr @Kernel11, i32 5620, i32 32}
; CHECK: !91 = !{ptr @Kernel11, i32 5620, i32 16}
; CHECK: !92 = !{ptr @Kernel11, i32 5622, i32 64}
; CHECK: !93 = !{ptr @Kernel11, i32 5622, i32 32}
; CHECK: !94 = !{ptr @Kernel11, i32 5622, i32 16}
; CHECK: !95 = !{ptr @Kernel11, i32 4459, i32 32}
attributes #11 = { "sycl-floating-point-control"="145" }

; rtz + denorm_allow + alt
; CHECK: !96 = !{ptr @Kernel12, i32 4463, i32 64}
; CHECK: !97 = !{ptr @Kernel12, i32 4463, i32 32}
; CHECK: !98 = !{ptr @Kernel12, i32 4463, i32 16}
; CHECK: !99 = !{ptr @Kernel12, i32 5622, i32 64}
; CHECK: !100 = !{ptr @Kernel12, i32 5622, i32 32}
; CHECK: !101 = !{ptr @Kernel12, i32 5622, i32 16}
; CHECK: !102 = !{ptr @Kernel12, i32 4459, i32 16}
; CHECK: !103 = !{ptr @Kernel12, i32 4459, i32 32}
; CHECK: !104 = !{ptr @Kernel12, i32 4459, i32 64}
attributes #12 = { "sycl-floating-point-control"="1265" }
