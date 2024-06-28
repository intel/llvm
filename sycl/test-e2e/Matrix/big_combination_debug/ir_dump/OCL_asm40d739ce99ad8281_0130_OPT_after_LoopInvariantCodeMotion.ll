; ------------------------------------------------
; OCL_asm40d739ce99ad8281_0130_OPT_after_LoopInvariantCodeMotion.ll
; LLVM major version: 14
; ------------------------------------------------

; Preheader:
_Z18get_sub_group_sizev.exit.i:                   ; preds = %__builtin_spriv_OpJointMatrixLoadINTEL_Accumulator_RowMajor_32x32_i32_128_global_v8i8_pi32_i32.23.exit
  %.fca.0.gep = getelementptr inbounds [2 x <64 x float>], [2 x <64 x float>]* %10, i64 0, i64 0
  store <64 x float> %56, <64 x float>* %.fca.0.gep, align 256
  %.fca.1.gep = getelementptr inbounds [2 x <64 x float>], [2 x <64 x float>]* %10, i64 0, i64 1
  store <64 x float> %58, <64 x float>* %.fca.1.gep, align 256
  %59 = bitcast [2 x <64 x float>]* %10 to i32*
  %60 = shl nuw nsw i32 %simdLaneId, 1
  %61 = and i32 %60, 131008
  %62 = or i32 %61, %38
  %63 = zext i32 %62 to i64
  %64 = getelementptr inbounds float, float addrspace(1)* %36, i64 %63
  %65 = bitcast float addrspace(1)* %64 to i32 addrspace(1)*
  br label %66

; Loop:
66:                                               ; preds = %._crit_edge98, %_Z18get_sub_group_sizev.exit.i
  %67 = phi i32 [ 0, %_Z18get_sub_group_sizev.exit.i ], [ %72, %._crit_edge98 ]
  br i1 %.lcssa, label %68, label %._crit_edge97

._crit_edge97:                                    ; preds = %66
  br label %._crit_edge98

68:                                               ; preds = %66
  %69 = zext i32 %67 to i64
  %70 = getelementptr inbounds i32, i32* %59, i64 %69
  %71 = load i32, i32* %70, align 4, !tbaa !521
  store i32 %71, i32 addrspace(1)* %65, align 4, !tbaa !521
  br label %._crit_edge98

._crit_edge98:                                    ; preds = %._crit_edge97, %68
  %72 = add nuw nsw i32 %67, 1
  %73 = icmp ult i32 %67, 127
  br i1 %73, label %66, label %__builtin_spriv_OpJointMatrixStoreINTEL_Accumulator_RowMajor_32x32_i32_128_global_pi64_v8i8.26.exit

; Exit blocks
__builtin_spriv_OpJointMatrixStoreINTEL_Accumulator_RowMajor_32x32_i32_128_global_pi64_v8i8.26.exit: ; preds = %._crit_edge98
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
  %40 = shl nuw nsw i32 %simdLaneId, 1
  %41 = and i32 %40, 131008
  %42 = or i32 %41, %38
  %43 = zext i32 %42 to i64
  %44 = getelementptr inbounds float, float addrspace(1)* %36, i64 %43
  %45 = bitcast float addrspace(1)* %44 to i32 addrspace(1)*
  br label %46

; Loop:
46:                                               ; preds = %._crit_edge96, %_Z18get_sub_group_sizev.exit.i4
  %47 = phi i32 [ 0, %_Z18get_sub_group_sizev.exit.i4 ], [ %53, %._crit_edge96 ]
  br i1 %39, label %48, label %._crit_edge

._crit_edge:                                      ; preds = %46
  br label %._crit_edge96

48:                                               ; preds = %46
  %49 = load i32, i32 addrspace(1)* %45, align 4, !tbaa !521
  br label %._crit_edge96

._crit_edge96:                                    ; preds = %._crit_edge, %48
  %50 = phi i32 [ %49, %48 ], [ 0, %._crit_edge ]
  %51 = zext i32 %47 to i64
  %52 = getelementptr inbounds i32, i32* %37, i64 %51
  store i32 %50, i32* %52, align 4, !tbaa !521
  %53 = add nuw nsw i32 %47, 1
  %54 = icmp ult i32 %47, 127
  br i1 %54, label %46, label %__builtin_spriv_OpJointMatrixLoadINTEL_Accumulator_RowMajor_32x32_i32_128_global_v8i8_pi32_i32.23.exit

; Exit blocks
__builtin_spriv_OpJointMatrixLoadINTEL_Accumulator_RowMajor_32x32_i32_128_global_v8i8_pi32_i32.23.exit: ; preds = %._crit_edge96
  %.lcssa = phi i1 [ %39, %._crit_edge96 ]
  %55 = getelementptr inbounds [2 x <64 x float>], [2 x <64 x float>]* %11, i64 0, i64 0
  %56 = load <64 x float>, <64 x float>* %55, align 256
  %57 = getelementptr inbounds [2 x <64 x float>], [2 x <64 x float>]* %11, i64 0, i64 1
  %58 = load <64 x float>, <64 x float>* %57, align 256
  br label %_Z18get_sub_group_sizev.exit.i
