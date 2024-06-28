; ------------------------------------------------
; OCL_asm40d739ce99ad8281_0168_OPT_after_Inlinerforalways_inlinefunctions.ll
; LLVM major version: 14
; ------------------------------------------------
; Function Attrs: convergent nounwind
define spir_kernel void @_ZTS7imatrixIfLm32ELm32ELm16EE(float addrspace(1)* align 4 %0, %"class.sycl::_V1::range"* byval(%"class.sycl::_V1::range") align 8 %1, %"class.sycl::_V1::range"* byval(%"class.sycl::_V1::range") align 8 %2, i64 %3, %"class.sycl::_V1::ext::oneapi::bfloat16" addrspace(1)* align 2 %4, %"class.sycl::_V1::range"* byval(%"class.sycl::_V1::range") align 8 %5, %"class.sycl::_V1::range"* byval(%"class.sycl::_V1::range") align 8 %6, %"class.sycl::_V1::ext::oneapi::bfloat16" addrspace(1)* align 2 %7, %"class.sycl::_V1::range"* byval(%"class.sycl::_V1::range") align 8 %8, %"class.sycl::_V1::range"* byval(%"class.sycl::_V1::range") align 8 %9, <8 x i32> %r0, <8 x i32> %payloadHeader, <3 x i32> %localSize, <3 x i32> %enqueuedLocalSize, i16 %localIdX, i16 %localIdY, i16 %localIdZ, i8* %privateBase, i64 %const_reg_qword, i64 %const_reg_qword1, i64 %const_reg_qword2, i64 %const_reg_qword3, i64 %const_reg_qword4, i64 %const_reg_qword5, i64 %const_reg_qword6, i64 %const_reg_qword7, i64 %const_reg_qword8, i64 %const_reg_qword9, i64 %const_reg_qword10, i64 %const_reg_qword11, i32 %bufferOffset, i32 %bufferOffset12, i32 %bufferOffset13) #0 {
_Z18get_sub_group_sizev.exit.i4:
  %payloadHeader.scalar = extractelement <8 x i32> %payloadHeader, i64 0
  %payloadHeader.scalar87 = extractelement <8 x i32> %payloadHeader, i64 1
  %enqueuedLocalSize.scalar = extractelement <3 x i32> %enqueuedLocalSize, i64 0
  %enqueuedLocalSize.scalar85 = extractelement <3 x i32> %enqueuedLocalSize, i64 1
  %r0.scalar78 = extractelement <8 x i32> %r0, i64 1
  %r0.scalar83 = extractelement <8 x i32> %r0, i64 6
  %10 = mul i32 %enqueuedLocalSize.scalar85, %r0.scalar83
  %localIdY15 = zext i16 %localIdY to i32
  %11 = add i32 %10, %localIdY15
  %12 = add i32 %11, %payloadHeader.scalar87
  %13 = icmp sgt i32 %12, -1
  call void @llvm.assume(i1 %13)
  %14 = mul i32 %enqueuedLocalSize.scalar, %r0.scalar78
  %localIdX21 = zext i16 %localIdX to i32
  %15 = add i32 %14, %localIdX21
  %16 = add i32 %15, %payloadHeader.scalar
  %17 = icmp sgt i32 %16, -1
  call void @llvm.assume(i1 %17)
  %simdLaneId16 = call i16 @llvm.genx.GenISA.simdLaneId()
  %18 = icmp ult i16 %simdLaneId16, 1024
  br i1 %18, label %._crit_edge98, label %._crit_edge98.127

._crit_edge98:                                    ; preds = %_Z18get_sub_group_sizev.exit.i4
  br label %._crit_edge98.127

._crit_edge98.127:                                ; preds = %_Z18get_sub_group_sizev.exit.i4, %._crit_edge98
  ret void
}

Printing <null> Function

Printing <null> Function
