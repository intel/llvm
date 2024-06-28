; ------------------------------------------------
; OCL_asm0418e209feebb096_0124_OPT_after_Inlinerforalways_inlinefunctions.ll
; LLVM major version: 14
; ------------------------------------------------
; Function Attrs: convergent nounwind
define spir_kernel void @_ZTS7imatrixIfLm32ELm32ELm16EE(<8 x i32> %r0, <8 x i32> %payloadHeader, <3 x i32> %enqueuedLocalSize, i16 %localIdX, i16 %localIdY, i16 %localIdZ) #0 {
  %payloadHeader.scalar = extractelement <8 x i32> %payloadHeader, i64 0
  %payloadHeader.scalar21 = extractelement <8 x i32> %payloadHeader, i64 1
  %enqueuedLocalSize.scalar = extractelement <3 x i32> %enqueuedLocalSize, i64 0
  %enqueuedLocalSize.scalar19 = extractelement <3 x i32> %enqueuedLocalSize, i64 1
  %r0.scalar12 = extractelement <8 x i32> %r0, i64 1
  %r0.scalar17 = extractelement <8 x i32> %r0, i64 6
  %1 = mul i32 %enqueuedLocalSize.scalar19, %r0.scalar17
  %localIdY2 = zext i16 %localIdY to i32
  %2 = add i32 %1, %localIdY2
  %3 = add i32 %2, %payloadHeader.scalar21
  %4 = icmp sgt i32 %3, -1
  call void @llvm.assume(i1 %4)
  %5 = mul i32 %enqueuedLocalSize.scalar, %r0.scalar12
  %localIdX8 = zext i16 %localIdX to i32
  %6 = add i32 %5, %localIdX8
  %7 = add i32 %6, %payloadHeader.scalar
  %8 = icmp sgt i32 %7, -1
  call void @llvm.assume(i1 %8)
  ret void
}

Printing <null> Function

Printing <null> Function
