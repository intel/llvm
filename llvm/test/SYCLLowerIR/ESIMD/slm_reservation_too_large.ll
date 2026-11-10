; This test checks that too large SLM reservation sizes are diagnosed instead
; of being silently truncated or overflowing.
;
; RUN: split-file %s %t
; RUN: not --crash opt < %t/single.ll -passes=LowerESIMD,lower-slm-reservation-calls -S 2>&1 | FileCheck %s --check-prefix=SINGLE
; RUN: not --crash opt < %t/total.ll -passes=LowerESIMD,lower-slm-reservation-calls -S 2>&1 | FileCheck %s --check-prefix=TOTAL

; SINGLE: SLM reservation size is too large, function single
; TOTAL: total SLM usage is too large

;--- single.ll
target triple = "spir64-unknown-unknown"

declare spir_func void @_Z16__esimd_slm_initj(i32)

define spir_kernel void @single() !sycl_explicit_simd !0 {
  call spir_func void @_Z16__esimd_slm_initj(i32 -2147483648)
  ret void
}

!0 = !{}

;--- total.ll
target triple = "spir64-unknown-unknown"

declare spir_func void @_Z16__esimd_slm_initj(i32)
declare spir_func i32 @_Z17__esimd_slm_allocj(i32)
declare spir_func void @_Z16__esimd_slm_freej(i32)

define spir_kernel void @total() !sycl_explicit_simd !0 {
  call spir_func void @_Z16__esimd_slm_initj(i32 2147483647)
  %slm = call spir_func i32 @_Z17__esimd_slm_allocj(i32 2147483647)
  call spir_func void @_Z16__esimd_slm_freej(i32 %slm)
  ret void
}

!0 = !{}
