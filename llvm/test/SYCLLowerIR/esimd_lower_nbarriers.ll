; RUN: opt < %s -LowerESIMD -S | FileCheck %s

target datalayout = "e-i64:64-v16:16-v24:32-v32:32-v48:64-v96:128-v192:256-v256:256-v512:512-v1024:1024-n8:16:32:64"
target triple = "spir64-unknown-unknown-sycldevice"

; Function Attrs: convergent norecurse mustprogress
define dso_local spir_kernel void @_ZTSZ6calleriE12kernel_esimd() !sycl_explicit_simd !3 {
entry:
; CHECK: call void @llvm.genx.nbarrier(i8 0, i8 2, i8 0)
  call spir_func void @_Z16__esimd_nbarrierhhh(i8 zeroext 0, i8 zeroext 2, i8 zeroext 0)

; CHECK: call void @llvm.genx.raw.send.noresult.i1.v8i32(i32 0, i1 true, i32 3, i32 33554436, <8 x i32> <i32 0, i32 0, i32 67371008, i32 0, i32 0, i32 0, i32 0, i32 0>)
  call spir_func void @_Z32__esimd_raw_send_nbarrier_signalIjLi8EEvjjjN2cl4sycl5INTEL3gpu6detail11vector_typeIT_XT0_EE4typeEt(i32 0, i32 3, i32 33554436, <8 x i32> <i32 0, i32 0, i32 67371008, i32 0, i32 0, i32
0, i32 0, i32 0>, i16 zeroext 1)

  ret void
}
!3 = !{}

declare dso_local spir_func void @_Z16__esimd_nbarrierhhh(i8 zeroext, i8 zeroext, i8 zeroext) local_unnamed_addr #1
declare dso_local spir_func void @_Z32__esimd_raw_send_nbarrier_signalIjLi8EEvjjjN2cl4sycl5INTEL3gpu6detail11vector_typeIT_XT0_EE4typeEt(i32, i32, i32, <8 x i32>, i16 zeroext)
