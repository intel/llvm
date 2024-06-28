; ------------------------------------------------
; OCL_asm40d739ce99ad8281_0123_OPT_after_Deletedeadloops.ll
; LLVM major version: 14
; ------------------------------------------------

; Preheader:
_Z18get_sub_group_sizev.exit.i:                   ; preds = %__builtin_spriv_OpJointMatrixLoadINTEL_Accumulator_RowMajor_32x32_i32_128_global_v8i8_pi32_i32.23.exit
  %.fca.0.gep = getelementptr inbounds [2 x <64 x float>], [2 x <64 x float>]* %10, i64 0, i64 0
  store <64 x float> %57, <64 x float>* %.fca.0.gep, align 256
  %.fca.1.gep = getelementptr inbounds [2 x <64 x float>], [2 x <64 x float>]* %10, i64 0, i64 1
  store <64 x float> %59, <64 x float>* %.fca.1.gep, align 256
  %60 = bitcast [2 x <64 x float>]* %10 to i32*
  br label %61

; Loop:
61:                                               ; preds = %73, %_Z18get_sub_group_sizev.exit.i
  %62 = phi i32 [ 0, %_Z18get_sub_group_sizev.exit.i ], [ %74, %73 ]
  br i1 %.lcssa, label %63, label %73

63:                                               ; preds = %61
  %64 = zext i32 %62 to i64
  %65 = getelementptr inbounds i32, i32* %60, i64 %64
  %66 = load i32, i32* %65, align 4, !tbaa !521
  %67 = shl nuw nsw i32 %simdLaneId, 1
  %68 = and i32 %67, 131008
  %69 = or i32 %68, %38
  %70 = zext i32 %69 to i64
  %71 = getelementptr inbounds float, float addrspace(1)* %36, i64 %70
  %72 = bitcast float addrspace(1)* %71 to i32 addrspace(1)*
  store i32 %66, i32 addrspace(1)* %72, align 4, !tbaa !521
  br label %73

73:                                               ; preds = %63, %61
  %74 = add nuw nsw i32 %62, 1
  %75 = icmp ult i32 %62, 127
  br i1 %75, label %61, label %__builtin_spriv_OpJointMatrixStoreINTEL_Accumulator_RowMajor_32x32_i32_128_global_pi64_v8i8.26.exit

; Exit blocks
__builtin_spriv_OpJointMatrixStoreINTEL_Accumulator_RowMajor_32x32_i32_128_global_pi64_v8i8.26.exit: ; preds = %73
  ret void

; Preheader:
_Z18get_sub_group_sizev.exit.i4:
  %payloadHeader.scalar = extractelement <8 x i32> %payloadHeader, i64 0
  %payloadHeader.scalar87 = extractelement <8 x i32> %payloadHeader, i64 1
  %enqueuedLocalSize.scalar = extractelement <3 x i32> %enqueuedLocalSize, i64 0
  %enqueuedLocalSize.scalar85 = extractelement <3 x i32> %enqueuedLocalSize, i64 1
  %r0.scalar78 = extractelement <8 x i32> %r0, i64 1
  %r0.scalar83 = extractelement <8 x i32> %r0, i64 6
  %10 = alloca [2 x <64 x float>], align 256
  %11 = alloca [2 x <64 x float>], align 256
  %12 = mul i64 %const_reg_qword2, %const_reg_qword1
  %13 = getelementptr float, float addrspace(1)* %0, i64 %12
  %14 = getelementptr float, float addrspace(1)* %13, i64 %const_reg_qword3
  %15 = mul i32 %enqueuedLocalSize.scalar85, %r0.scalar83
  %localIdY15 = zext i16 %localIdY to i32
  %16 = add i32 %15, %localIdY15
  %17 = add i32 %16, %payloadHeader.scalar87
  %18 = zext i32 %17 to i64
  %19 = icmp sgt i32 %17, -1
  call void @llvm.assume(i1 %19)
  %20 = mul i32 %enqueuedLocalSize.scalar, %r0.scalar78
  %localIdX21 = zext i16 %localIdX to i32
  %21 = add i32 %20, %localIdX21
  %22 = add i32 %21, %payloadHeader.scalar
  %23 = zext i32 %22 to i64
  %24 = icmp sgt i32 %22, -1
  call void @llvm.assume(i1 %24)
  %25 = zext i16 %localIdY to i64
  %26 = sub nsw i64 %18, %25, !spirv.Decorations !519
  %27 = zext i16 %localIdX to i64
  %28 = sub nsw i64 %23, %27, !spirv.Decorations !519
  %29 = add i64 %12, %const_reg_qword3
  %30 = sub i64 0, %29
  %31 = getelementptr inbounds float, float addrspace(1)* %14, i64 %30
  %32 = shl nsw i64 %26, 11, !spirv.Decorations !519
  %33 = getelementptr inbounds float, float addrspace(1)* %31, i64 %32
  %34 = udiv i64 %28, %3
  %freeze = freeze i64 %34
  %35 = shl i64 %freeze, 5
  %36 = getelementptr inbounds float, float addrspace(1)* %33, i64 %35
  %simdLaneId16 = call i16 @llvm.genx.GenISA.simdLaneId()
  %simdLaneId16.fr = freeze i16 %simdLaneId16
  %simdLaneId = zext i16 %simdLaneId16.fr to i32
  %37 = bitcast [2 x <64 x float>]* %11 to i32*
  %38 = and i32 %simdLaneId, 31
  %39 = icmp ult i16 %simdLaneId16.fr, 1024
  br label %40

; Loop:
40:                                               ; preds = %50, %_Z18get_sub_group_sizev.exit.i4
  %41 = phi i32 [ 0, %_Z18get_sub_group_sizev.exit.i4 ], [ %54, %50 ]
  br i1 %39, label %42, label %50

42:                                               ; preds = %40
  %43 = shl nuw nsw i32 %simdLaneId, 1
  %44 = and i32 %43, 131008
  %45 = or i32 %44, %38
  %46 = zext i32 %45 to i64
  %47 = getelementptr inbounds float, float addrspace(1)* %36, i64 %46
  %48 = bitcast float addrspace(1)* %47 to i32 addrspace(1)*
  %49 = load i32, i32 addrspace(1)* %48, align 4, !tbaa !521
  br label %50

50:                                               ; preds = %42, %40
  %51 = phi i32 [ %49, %42 ], [ 0, %40 ]
  %52 = zext i32 %41 to i64
  %53 = getelementptr inbounds i32, i32* %37, i64 %52
  store i32 %51, i32* %53, align 4, !tbaa !521
  %54 = add nuw nsw i32 %41, 1
  %55 = icmp ult i32 %41, 127
  br i1 %55, label %40, label %__builtin_spriv_OpJointMatrixLoadINTEL_Accumulator_RowMajor_32x32_i32_128_global_v8i8_pi32_i32.23.exit

; Exit blocks
__builtin_spriv_OpJointMatrixLoadINTEL_Accumulator_RowMajor_32x32_i32_128_global_v8i8_pi32_i32.23.exit: ; preds = %50
  %.lcssa = phi i1 [ %39, %50 ]
  %56 = getelementptr inbounds [2 x <64 x float>], [2 x <64 x float>]* %11, i64 0, i64 0
  %57 = load <64 x float>, <64 x float>* %56, align 256
  %58 = getelementptr inbounds [2 x <64 x float>], [2 x <64 x float>]* %11, i64 0, i64 1
  %59 = load <64 x float>, <64 x float>* %58, align 256
  br label %_Z18get_sub_group_sizev.exit.i
