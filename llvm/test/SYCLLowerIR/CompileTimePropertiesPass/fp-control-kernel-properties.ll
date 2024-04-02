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

; SPIRV execution modes for FP control.   | BitMask
; ROUNDING_MODE_RTE = 4462;               | 00000001
; ROUNDING_MODE_RTP_INTEL = 5620;         | 00000010
; ROUNDING_MODE_RTN_INTEL = 5621;         | 00000100
; ROUNDING_MODE_RTZ = 4463;               | 00001000
; DEMORM_FLUSH_TO_ZERO = 4460;            | 00010000
; DENORM_PRESERVE (double) = 4459;        | 00100000
; DENORM_PRESERVE (float) = 4459;         | 01000000
; DENORM_PRESERVE (half) = 4459;          | 10000000

; rte + ftz (Default)
; CHECK: !0 = !{ptr @Kernel0, i32 [[RTE:4462]], i32 64}
; CHECK: !1 = !{ptr @Kernel0, i32 [[RTE]], i32 32}
; CHECK: !2 = !{ptr @Kernel0, i32 [[RTE]], i32 16}
; CHECK: !3 = !{ptr @Kernel0, i32 [[FTZ:4460]], i32 64}
; CHECK: !4 = !{ptr @Kernel0, i32 [[FTZ]], i32 32}
; CHECK: !5 = !{ptr @Kernel0, i32 [[FTZ]], i32 16}
attributes #0 = { "sycl-floating-point-control"="17" }

; rtp + ftz
; CHECK: !6 = !{ptr @Kernel1, i32 [[RTP:5620]], i32 64}
; CHECK: !7 = !{ptr @Kernel1, i32 [[RTP]], i32 32}
; CHECK: !8 = !{ptr @Kernel1, i32 [[RTP]], i32 16}
; CHECK: !9 = !{ptr @Kernel1, i32 [[FTZ]], i32 64}
; CHECK: !10 = !{ptr @Kernel1, i32 [[FTZ]], i32 32}
; CHECK: !11 = !{ptr @Kernel1, i32 [[FTZ]], i32 16}
attributes #1 = { "sycl-floating-point-control"="18" }

; rtn + ftz
; CHECK: !12 = !{ptr @Kernel2, i32 [[RTN:5621]], i32 64}
; CHECK: !13 = !{ptr @Kernel2, i32 [[RTN]], i32 32}
; CHECK: !14 = !{ptr @Kernel2, i32 [[RTN]], i32 16}
; CHECK: !15 = !{ptr @Kernel2, i32 [[FTZ]], i32 64}
; CHECK: !16 = !{ptr @Kernel2, i32 [[FTZ]], i32 32}
; CHECK: !17 = !{ptr @Kernel2, i32 [[FTZ]], i32 16}
attributes #2 = { "sycl-floating-point-control"="20" }

; rtz + ftz
; CHECK: !18 = !{ptr @Kernel3, i32 [[RTZ:4463]], i32 64}
; CHECK: !19 = !{ptr @Kernel3, i32 [[RTZ]], i32 32}
; CHECK: !20 = !{ptr @Kernel3, i32 [[RTZ]], i32 16}
; CHECK: !21 = !{ptr @Kernel3, i32 [[FTZ]], i32 64}
; CHECK: !22 = !{ptr @Kernel3, i32 [[FTZ]], i32 32}
; CHECK: !23 = !{ptr @Kernel3, i32 [[FTZ]], i32 16}
attributes #3 = { "sycl-floating-point-control"="24" }

; rte + denorm_preserve(double)
; CHECK: !24 = !{ptr @Kernel4, i32 [[RTE]], i32 64}
; CHECK: !25 = !{ptr @Kernel4, i32 [[RTE]], i32 32}
; CHECK: !26 = !{ptr @Kernel4, i32 [[RTE]], i32 16}
; CHECK: !27 = !{ptr @Kernel4, i32 [[DENORM_PRESERVE:4459]], i32 64}
attributes #4 = { "sycl-floating-point-control"="33" }

; rte + denorm_preserve(float)
; CHECK: !28 = !{ptr @Kernel5, i32 [[RTE]], i32 64}
; CHECK: !29 = !{ptr @Kernel5, i32 [[RTE]], i32 32}
; CHECK: !30 = !{ptr @Kernel5, i32 [[RTE]], i32 16}
; CHECK: !31 = !{ptr @Kernel5, i32 [[DENORM_PRESERVE]], i32 32}
attributes #5 = { "sycl-floating-point-control"="65" }

; rte + denorm_preserve(half)
; CHECK: !32 = !{ptr @Kernel6, i32 [[RTE]], i32 64}
; CHECK: !33 = !{ptr @Kernel6, i32 [[RTE]], i32 32}
; CHECK: !34 = !{ptr @Kernel6, i32 [[RTE]], i32 16}
; CHECK: !35 = !{ptr @Kernel6, i32 [[DENORM_PRESERVE]], i32 16}
attributes #6 = { "sycl-floating-point-control"="129" }

; rte + denorm_allow
; CHECK: !36 = !{ptr @Kernel7, i32 [[RTE]], i32 64}
; CHECK: !37 = !{ptr @Kernel7, i32 [[RTE]], i32 32}
; CHECK: !38 = !{ptr @Kernel7, i32 [[RTE]], i32 16}
; CHECK: !39 = !{ptr @Kernel7, i32 [[DENORM_PRESERVE]], i32 16}
; CHECK: !40 = !{ptr @Kernel7, i32 [[DENORM_PRESERVE]], i32 32}
; CHECK: !41 = !{ptr @Kernel7, i32 [[DENORM_PRESERVE]], i32 64}
attributes #7 = { "sycl-floating-point-control"="225" }

; rtz + denorm_preserve(double)
; CHECK: !42 = !{ptr @Kernel8, i32 [[RTZ]], i32 64}
; CHECK: !43 = !{ptr @Kernel8, i32 [[RTZ]], i32 32}
; CHECK: !44 = !{ptr @Kernel8, i32 [[RTZ]], i32 16}
; CHECK: !45 = !{ptr @Kernel8, i32 [[DENORM_PRESERVE]], i32 64}
attributes #8 = { "sycl-floating-point-control"="40" }

; rtp + denorm_preserve(float)
; CHECK: !46 = !{ptr @Kernel9, i32 [[RTP]], i32 64}
; CHECK: !47 = !{ptr @Kernel9, i32 [[RTP]], i32 32}
; CHECK: !48 = !{ptr @Kernel9, i32 [[RTP]], i32 16}
; CHECK: !49 = !{ptr @Kernel9, i32 [[DENORM_PRESERVE]], i32 32}
attributes #9 = { "sycl-floating-point-control"="66" }

; rtz + denorm_allow
; CHECK: !50 = !{ptr @Kernel10, i32 [[RTZ]], i32 64}
; CHECK: !51 = !{ptr @Kernel10, i32 [[RTZ]], i32 32}
; CHECK: !52 = !{ptr @Kernel10, i32 [[RTZ]], i32 16}
; CHECK: !53 = !{ptr @Kernel10, i32 [[DENORM_PRESERVE]], i32 16}
; CHECK: !54 = !{ptr @Kernel10, i32 [[DENORM_PRESERVE]], i32 32}
; CHECK: !55 = !{ptr @Kernel10, i32 [[DENORM_PRESERVE]], i32 64}
attributes #10 = { "sycl-floating-point-control"="232" }
