; ------------------------------------------------
; OCL_asm40d739ce99ad8281_0093_Unify_after_Combineredundantinstructions.ll
; LLVM major version: 14
; ------------------------------------------------
target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v16:16:16-v24:32:32-v32:32:32-v48:64:64-v64:64:64-v96:128:128-v128:128:128-v192:256:256-v256:256:256-v512:512:512-v1024:1024:1024-n8:16:32"
target triple = "spir64-unknown-unknown"

%"class.sycl::_V1::ext::oneapi::bfloat16" = type { i16 }
%"class.sycl::_V1::range" = type { %"class.sycl::_V1::detail::array" }
%"class.sycl::_V1::detail::array" = type { [2 x i64] }

; Function Attrs: convergent nounwind
define spir_kernel void @_ZTS7imatrixIfLm32ELm32ELm16EE(float addrspace(1)* align 4 %0, %"class.sycl::_V1::range"* byval(%"class.sycl::_V1::range") align 8 %1, %"class.sycl::_V1::range"* byval(%"class.sycl::_V1::range") align 8 %2, i64 %3, %"class.sycl::_V1::ext::oneapi::bfloat16" addrspace(1)* align 2 %4, %"class.sycl::_V1::range"* byval(%"class.sycl::_V1::range") align 8 %5, %"class.sycl::_V1::range"* byval(%"class.sycl::_V1::range") align 8 %6, %"class.sycl::_V1::ext::oneapi::bfloat16" addrspace(1)* align 2 %7, %"class.sycl::_V1::range"* byval(%"class.sycl::_V1::range") align 8 %8, %"class.sycl::_V1::range"* byval(%"class.sycl::_V1::range") align 8 %9, <8 x i32> %r0, <8 x i32> %payloadHeader, <3 x i32> %localSize, <3 x i32> %enqueuedLocalSize, i16 %localIdX, i16 %localIdY, i16 %localIdZ, i8* %privateBase, i64 %const_reg_qword, i64 %const_reg_qword1, i64 %const_reg_qword2, i64 %const_reg_qword3, i64 %const_reg_qword4, i64 %const_reg_qword5, i64 %const_reg_qword6, i64 %const_reg_qword7, i64 %const_reg_qword8, i64 %const_reg_qword9, i64 %const_reg_qword10, i64 %const_reg_qword11, i32 %bufferOffset, i32 %bufferOffset12, i32 %bufferOffset13) #0 {
  %11 = alloca [2 x <64 x float>], align 256
  %12 = alloca [2 x <64 x float>], align 256
  %13 = mul i64 %const_reg_qword2, %const_reg_qword1
  %14 = getelementptr float, float addrspace(1)* %0, i64 %13
  %15 = getelementptr float, float addrspace(1)* %14, i64 %const_reg_qword3
  %groupId = extractelement <8 x i32> %r0, i64 6
  %enqueuedLocalSize14 = extractelement <3 x i32> %enqueuedLocalSize, i64 1
  %16 = mul i32 %enqueuedLocalSize14, %groupId
  %localIdY15 = zext i16 %localIdY to i32
  %17 = add i32 %16, %localIdY15
  %globalOffset = extractelement <8 x i32> %payloadHeader, i64 1
  %18 = add i32 %17, %globalOffset
  %19 = zext i32 %18 to i64
  %20 = icmp sgt i32 %18, -1
  call void @llvm.assume(i1 %20)
  %groupId19 = extractelement <8 x i32> %r0, i64 1
  %enqueuedLocalSize20 = extractelement <3 x i32> %enqueuedLocalSize, i64 0
  %21 = mul i32 %enqueuedLocalSize20, %groupId19
  %localIdX21 = zext i16 %localIdX to i32
  %22 = add i32 %21, %localIdX21
  %globalOffset22 = extractelement <8 x i32> %payloadHeader, i64 0
  %23 = add i32 %22, %globalOffset22
  %24 = zext i32 %23 to i64
  %25 = icmp sgt i32 %23, -1
  call void @llvm.assume(i1 %25)
  %26 = zext i16 %localIdY to i64
  %27 = sub nsw i64 %19, %26, !spirv.Decorations !516
  %28 = zext i16 %localIdX to i64
  %29 = sub nsw i64 %24, %28, !spirv.Decorations !516
  %30 = add i64 %13, %const_reg_qword3
  %31 = sub i64 0, %30
  %32 = getelementptr inbounds float, float addrspace(1)* %15, i64 %31
  %33 = shl nsw i64 %27, 11, !spirv.Decorations !516
  %34 = getelementptr inbounds float, float addrspace(1)* %32, i64 %33
  %35 = udiv i64 %29, %3
  %freeze = freeze i64 %35
  %36 = shl i64 %freeze, 5
  %37 = getelementptr inbounds float, float addrspace(1)* %34, i64 %36
  %localSize25 = extractelement <3 x i32> %localSize, i64 0
  %localSize26 = extractelement <3 x i32> %localSize, i64 1
  %38 = mul i32 %localSize26, %localSize25
  %localSize27 = extractelement <3 x i32> %localSize, i64 2
  %39 = mul i32 %38, %localSize27
  %40 = and i32 %39, 7
  %41 = icmp eq i32 %40, 0
  br i1 %41, label %54, label %42

42:                                               ; preds = %10
  %localIdZ28 = zext i16 %localIdZ to i32
  %localSize29 = extractelement <3 x i32> %localSize, i64 1
  %43 = mul i32 %localSize29, %localIdZ28
  %localIdY30 = zext i16 %localIdY to i32
  %44 = add i32 %43, %localIdY30
  %localSize31 = extractelement <3 x i32> %localSize, i64 0
  %45 = mul i32 %44, %localSize31
  %localIdX32 = zext i16 %localIdX to i32
  %46 = add i32 %45, %localIdX32
  %47 = lshr i32 %46, 3
  %localSize33 = extractelement <3 x i32> %localSize, i64 0
  %localSize34 = extractelement <3 x i32> %localSize, i64 1
  %48 = mul i32 %localSize34, %localSize33
  %localSize35 = extractelement <3 x i32> %localSize, i64 2
  %49 = mul i32 %48, %localSize35
  %50 = add i32 %49, 7
  %51 = lshr i32 %50, 3
  %52 = add nsw i32 %51, -1
  %53 = icmp ult i32 %47, %52
  br i1 %53, label %54, label %_Z18get_sub_group_sizev.exit.i4

54:                                               ; preds = %42, %10
  br label %_Z18get_sub_group_sizev.exit.i4

_Z18get_sub_group_sizev.exit.i4:                  ; preds = %54, %42
  %simdLaneId16 = call i16 @llvm.genx.GenISA.simdLaneId()
  %simdLaneId16.fr = freeze i16 %simdLaneId16
  %simdLaneId = zext i16 %simdLaneId16.fr to i32
  %55 = bitcast [2 x <64 x float>]* %12 to i32*
  %56 = and i32 %simdLaneId, 31
  br label %57

57:                                               ; preds = %68, %_Z18get_sub_group_sizev.exit.i4
  %58 = phi i32 [ 0, %_Z18get_sub_group_sizev.exit.i4 ], [ %72, %68 ]
  %59 = icmp ult i16 %simdLaneId16.fr, 1024
  br i1 %59, label %60, label %68

60:                                               ; preds = %57
  %61 = shl nuw nsw i32 %simdLaneId, 1
  %62 = and i32 %61, 131008
  %63 = or i32 %62, %56
  %64 = zext i32 %63 to i64
  %65 = getelementptr inbounds float, float addrspace(1)* %37, i64 %64
  %66 = bitcast float addrspace(1)* %65 to i32 addrspace(1)*
  %67 = load i32, i32 addrspace(1)* %66, align 4, !tbaa !518
  br label %68

68:                                               ; preds = %60, %57
  %69 = phi i32 [ %67, %60 ], [ 0, %57 ]
  %70 = zext i32 %58 to i64
  %71 = getelementptr inbounds i32, i32* %55, i64 %70
  store i32 %69, i32* %71, align 4, !tbaa !518
  %72 = add nuw nsw i32 %58, 1
  %73 = icmp ult i32 %58, 127
  br i1 %73, label %57, label %__builtin_spriv_OpJointMatrixLoadINTEL_Accumulator_RowMajor_32x32_i32_128_global_v8i8_pi32_i32.23.exit

__builtin_spriv_OpJointMatrixLoadINTEL_Accumulator_RowMajor_32x32_i32_128_global_v8i8_pi32_i32.23.exit: ; preds = %68
  %74 = getelementptr inbounds [2 x <64 x float>], [2 x <64 x float>]* %12, i64 0, i64 0
  %75 = load <64 x float>, <64 x float>* %74, align 256
  %76 = getelementptr inbounds [2 x <64 x float>], [2 x <64 x float>]* %12, i64 0, i64 1
  %77 = load <64 x float>, <64 x float>* %76, align 256
  br label %78

78:                                               ; preds = %__builtin_spriv_OpJointMatrixLoadINTEL_PackedB_PackedB_16x32_i16_32_global_v8i8_pi32_i32.25.exit, %__builtin_spriv_OpJointMatrixLoadINTEL_Accumulator_RowMajor_32x32_i32_128_global_v8i8_pi32_i32.23.exit
  %79 = phi i32 [ 0, %__builtin_spriv_OpJointMatrixLoadINTEL_Accumulator_RowMajor_32x32_i32_128_global_v8i8_pi32_i32.23.exit ], [ %127, %__builtin_spriv_OpJointMatrixLoadINTEL_PackedB_PackedB_16x32_i16_32_global_v8i8_pi32_i32.25.exit ]
  %80 = icmp ult i32 %79, 2
  br i1 %80, label %81, label %128

81:                                               ; preds = %78
  %localSize36 = extractelement <3 x i32> %localSize, i64 0
  %localSize37 = extractelement <3 x i32> %localSize, i64 1
  %82 = mul i32 %localSize37, %localSize36
  %localSize38 = extractelement <3 x i32> %localSize, i64 2
  %83 = mul i32 %82, %localSize38
  %84 = and i32 %83, 7
  %85 = icmp eq i32 %84, 0
  br i1 %85, label %98, label %86

86:                                               ; preds = %81
  %localIdZ39 = zext i16 %localIdZ to i32
  %localSize40 = extractelement <3 x i32> %localSize, i64 1
  %87 = mul i32 %localSize40, %localIdZ39
  %localIdY41 = zext i16 %localIdY to i32
  %88 = add i32 %87, %localIdY41
  %localSize42 = extractelement <3 x i32> %localSize, i64 0
  %89 = mul i32 %88, %localSize42
  %localIdX43 = zext i16 %localIdX to i32
  %90 = add i32 %89, %localIdX43
  %91 = lshr i32 %90, 3
  %localSize44 = extractelement <3 x i32> %localSize, i64 0
  %localSize45 = extractelement <3 x i32> %localSize, i64 1
  %92 = mul i32 %localSize45, %localSize44
  %localSize46 = extractelement <3 x i32> %localSize, i64 2
  %93 = mul i32 %92, %localSize46
  %94 = add i32 %93, 7
  %95 = lshr i32 %94, 3
  %96 = add nsw i32 %95, -1
  %97 = icmp ult i32 %91, %96
  br i1 %97, label %98, label %_Z18get_sub_group_sizev.exit.i6

98:                                               ; preds = %86, %81
  br label %_Z18get_sub_group_sizev.exit.i6

_Z18get_sub_group_sizev.exit.i6:                  ; preds = %98, %86
  br label %99

99:                                               ; preds = %99, %_Z18get_sub_group_sizev.exit.i6
  %100 = phi i32 [ 0, %_Z18get_sub_group_sizev.exit.i6 ], [ %101, %99 ]
  %101 = add nuw nsw i32 %100, 1
  %102 = icmp ult i32 %100, 31
  br i1 %102, label %99, label %__builtin_spriv_OpJointMatrixLoadINTEL_PackedA_RowMajor_32x16_i16_32_global_v8i8_pi32_i32.24.exit

__builtin_spriv_OpJointMatrixLoadINTEL_PackedA_RowMajor_32x16_i16_32_global_v8i8_pi32_i32.24.exit: ; preds = %99
  %localSize47 = extractelement <3 x i32> %localSize, i64 0
  %localSize48 = extractelement <3 x i32> %localSize, i64 1
  %103 = mul i32 %localSize48, %localSize47
  %localSize49 = extractelement <3 x i32> %localSize, i64 2
  %104 = mul i32 %103, %localSize49
  %105 = and i32 %104, 7
  %106 = icmp eq i32 %105, 0
  br i1 %106, label %119, label %107

107:                                              ; preds = %__builtin_spriv_OpJointMatrixLoadINTEL_PackedA_RowMajor_32x16_i16_32_global_v8i8_pi32_i32.24.exit
  %localIdZ50 = zext i16 %localIdZ to i32
  %localSize51 = extractelement <3 x i32> %localSize, i64 1
  %108 = mul i32 %localSize51, %localIdZ50
  %localIdY52 = zext i16 %localIdY to i32
  %109 = add i32 %108, %localIdY52
  %localSize53 = extractelement <3 x i32> %localSize, i64 0
  %110 = mul i32 %109, %localSize53
  %localIdX54 = zext i16 %localIdX to i32
  %111 = add i32 %110, %localIdX54
  %112 = lshr i32 %111, 3
  %localSize55 = extractelement <3 x i32> %localSize, i64 0
  %localSize56 = extractelement <3 x i32> %localSize, i64 1
  %113 = mul i32 %localSize56, %localSize55
  %localSize57 = extractelement <3 x i32> %localSize, i64 2
  %114 = mul i32 %113, %localSize57
  %115 = add i32 %114, 7
  %116 = lshr i32 %115, 3
  %117 = add nsw i32 %116, -1
  %118 = icmp ult i32 %112, %117
  br i1 %118, label %119, label %_Z18get_sub_group_sizev.exit.i8

119:                                              ; preds = %107, %__builtin_spriv_OpJointMatrixLoadINTEL_PackedA_RowMajor_32x16_i16_32_global_v8i8_pi32_i32.24.exit
  br label %_Z18get_sub_group_sizev.exit.i8

_Z18get_sub_group_sizev.exit.i8:                  ; preds = %119, %107
  %simdLaneId1669 = call i16 @llvm.genx.GenISA.simdLaneId()
  %simdLaneId1669.fr = freeze i16 %simdLaneId1669
  br label %120

120:                                              ; preds = %124, %_Z18get_sub_group_sizev.exit.i8
  %121 = phi i32 [ 0, %_Z18get_sub_group_sizev.exit.i8 ], [ %125, %124 ]
  %122 = icmp ult i16 %simdLaneId1669.fr, 256
  br i1 %122, label %123, label %124

123:                                              ; preds = %120
  br label %124

124:                                              ; preds = %123, %120
  %125 = add nuw nsw i32 %121, 1
  %126 = icmp ult i32 %121, 31
  br i1 %126, label %120, label %__builtin_spriv_OpJointMatrixLoadINTEL_PackedB_PackedB_16x32_i16_32_global_v8i8_pi32_i32.25.exit

__builtin_spriv_OpJointMatrixLoadINTEL_PackedB_PackedB_16x32_i16_32_global_v8i8_pi32_i32.25.exit: ; preds = %124
  %127 = add nuw nsw i32 %79, 1, !spirv.Decorations !522
  br label %78

128:                                              ; preds = %78
  %.fca.0.gep = getelementptr inbounds [2 x <64 x float>], [2 x <64 x float>]* %11, i64 0, i64 0
  store <64 x float> %75, <64 x float>* %.fca.0.gep, align 256
  %.fca.1.gep = getelementptr inbounds [2 x <64 x float>], [2 x <64 x float>]* %11, i64 0, i64 1
  store <64 x float> %77, <64 x float>* %.fca.1.gep, align 256
  %localSize58 = extractelement <3 x i32> %localSize, i64 0
  %localSize59 = extractelement <3 x i32> %localSize, i64 1
  %129 = mul i32 %localSize59, %localSize58
  %localSize60 = extractelement <3 x i32> %localSize, i64 2
  %130 = mul i32 %129, %localSize60
  %131 = and i32 %130, 7
  %132 = icmp eq i32 %131, 0
  br i1 %132, label %145, label %133

133:                                              ; preds = %128
  %localIdZ61 = zext i16 %localIdZ to i32
  %localSize62 = extractelement <3 x i32> %localSize, i64 1
  %134 = mul i32 %localSize62, %localIdZ61
  %localIdY63 = zext i16 %localIdY to i32
  %135 = add i32 %134, %localIdY63
  %localSize64 = extractelement <3 x i32> %localSize, i64 0
  %136 = mul i32 %135, %localSize64
  %localIdX65 = zext i16 %localIdX to i32
  %137 = add i32 %136, %localIdX65
  %138 = lshr i32 %137, 3
  %localSize66 = extractelement <3 x i32> %localSize, i64 0
  %localSize67 = extractelement <3 x i32> %localSize, i64 1
  %139 = mul i32 %localSize67, %localSize66
  %localSize68 = extractelement <3 x i32> %localSize, i64 2
  %140 = mul i32 %139, %localSize68
  %141 = add i32 %140, 7
  %142 = lshr i32 %141, 3
  %143 = add nsw i32 %142, -1
  %144 = icmp ult i32 %138, %143
  br i1 %144, label %145, label %_Z18get_sub_group_sizev.exit.i

145:                                              ; preds = %133, %128
  br label %_Z18get_sub_group_sizev.exit.i

_Z18get_sub_group_sizev.exit.i:                   ; preds = %145, %133
  %simdLaneId1671 = call i16 @llvm.genx.GenISA.simdLaneId()
  %simdLaneId1671.fr = freeze i16 %simdLaneId1671
  %simdLaneId72 = zext i16 %simdLaneId1671.fr to i32
  %146 = bitcast [2 x <64 x float>]* %11 to i32*
  %147 = and i32 %simdLaneId72, 31
  br label %148

148:                                              ; preds = %161, %_Z18get_sub_group_sizev.exit.i
  %149 = phi i32 [ 0, %_Z18get_sub_group_sizev.exit.i ], [ %162, %161 ]
  %150 = icmp ult i16 %simdLaneId1671.fr, 1024
  br i1 %150, label %151, label %161

151:                                              ; preds = %148
  %152 = zext i32 %149 to i64
  %153 = getelementptr inbounds i32, i32* %146, i64 %152
  %154 = load i32, i32* %153, align 4, !tbaa !518
  %155 = shl nuw nsw i32 %simdLaneId72, 1
  %156 = and i32 %155, 131008
  %157 = or i32 %156, %147
  %158 = zext i32 %157 to i64
  %159 = getelementptr inbounds float, float addrspace(1)* %37, i64 %158
  %160 = bitcast float addrspace(1)* %159 to i32 addrspace(1)*
  store i32 %154, i32 addrspace(1)* %160, align 4, !tbaa !518
  br label %161

161:                                              ; preds = %151, %148
  %162 = add nuw nsw i32 %149, 1
  %163 = icmp ult i32 %149, 127
  br i1 %163, label %148, label %__builtin_spriv_OpJointMatrixStoreINTEL_Accumulator_RowMajor_32x32_i32_128_global_pi64_v8i8.26.exit

__builtin_spriv_OpJointMatrixStoreINTEL_Accumulator_RowMajor_32x32_i32_128_global_pi64_v8i8.26.exit: ; preds = %161
  ret void
}

; Function Attrs: inaccessiblememonly nofree nosync nounwind willreturn
declare void @llvm.assume(i1 noundef) #1

; Function Attrs: convergent
declare spir_func i32 @__builtin_IB_get_simd_size() local_unnamed_addr #2

; Function Attrs: convergent
declare spir_func i32 @__builtin_IB_get_simd_id() local_unnamed_addr #2

; Function Attrs: convergent
declare spir_func i32 @__builtin_IB_simd_block_read_1_global(i32 addrspace(1)* noundef) local_unnamed_addr #2

; Function Attrs: convergent mustprogress nofree nounwind readnone willreturn
declare spir_func i32 @__builtin_IB_get_local_size(i32 noundef) local_unnamed_addr #3

; Function Attrs: convergent mustprogress nofree nounwind readnone willreturn
declare spir_func i32 @__builtin_IB_get_local_id_z() local_unnamed_addr #3

; Function Attrs: convergent mustprogress nofree nounwind readnone willreturn
declare spir_func i32 @__builtin_IB_get_local_id_y() local_unnamed_addr #3

; Function Attrs: convergent mustprogress nofree nounwind readnone willreturn
declare spir_func i32 @__builtin_IB_get_local_id_x() local_unnamed_addr #3

; Function Attrs: convergent mustprogress nofree nounwind readnone willreturn
declare spir_func i32 @__builtin_IB_get_group_id(i32 noundef) local_unnamed_addr #3

; Function Attrs: convergent mustprogress nofree nounwind readnone willreturn
declare spir_func i32 @__builtin_IB_get_enqueued_local_size(i32 noundef) local_unnamed_addr #3

; Function Attrs: convergent mustprogress nofree nounwind readnone willreturn
declare spir_func i32 @__builtin_IB_get_global_offset(i32 noundef) local_unnamed_addr #3

; Function Attrs: nounwind readnone
declare i16 @llvm.genx.GenISA.simdLaneId() #4

; Function Attrs: nounwind readonly
declare i32 @llvm.genx.GenISA.simdBlockRead.i32.p1i32(i32 addrspace(1)*) #5

attributes #0 = { convergent nounwind "less-precise-fpmad"="true" }
attributes #1 = { inaccessiblememonly nofree nosync nounwind willreturn }
attributes #2 = { convergent "frame-pointer"="none" "no-trapping-math"="true" "stack-protector-buffer-size"="8" }
attributes #3 = { convergent mustprogress nofree nounwind readnone willreturn "frame-pointer"="none" "no-trapping-math"="true" "stack-protector-buffer-size"="8" }
attributes #4 = { nounwind readnone }
attributes #5 = { nounwind readonly }

!spirv.MemoryModel = !{!0}
!spirv.Source = !{!1}
!spirv.Generator = !{!2}
!igc.functions = !{!3}
!IGCMetadata = !{!42}
!opencl.ocl.version = !{!513, !513, !513, !513, !513}
!opencl.spir.version = !{!513, !513, !513, !513, !513}
!llvm.ident = !{!514, !514, !514, !514, !514}
!llvm.module.flags = !{!515}

!0 = !{i32 2, i32 2}
!1 = !{i32 4, i32 100000}
!2 = !{i16 6, i16 14}
!3 = !{void (float addrspace(1)*, %"class.sycl::_V1::range"*, %"class.sycl::_V1::range"*, i64, %"class.sycl::_V1::ext::oneapi::bfloat16" addrspace(1)*, %"class.sycl::_V1::range"*, %"class.sycl::_V1::range"*, %"class.sycl::_V1::ext::oneapi::bfloat16" addrspace(1)*, %"class.sycl::_V1::range"*, %"class.sycl::_V1::range"*, <8 x i32>, <8 x i32>, <3 x i32>, <3 x i32>, i16, i16, i16, i8*, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i32, i32, i32)* @_ZTS7imatrixIfLm32ELm32ELm16EE, !4}
!4 = !{!5, !6, !41}
!5 = !{!"function_type", i32 0}
!6 = !{!"implicit_arg_desc", !7, !8, !9, !10, !11, !12, !13, !14, !15, !18, !20, !22, !23, !25, !26, !28, !29, !31, !32, !34, !35, !37, !39}
!7 = !{i32 0}
!8 = !{i32 1}
!9 = !{i32 5}
!10 = !{i32 6}
!11 = !{i32 7}
!12 = !{i32 8}
!13 = !{i32 9}
!14 = !{i32 12}
!15 = !{i32 16, !16, !17}
!16 = !{!"explicit_arg_num", i32 1}
!17 = !{!"struct_arg_offset", i32 0}
!18 = !{i32 16, !16, !19}
!19 = !{!"struct_arg_offset", i32 8}
!20 = !{i32 16, !21, !17}
!21 = !{!"explicit_arg_num", i32 2}
!22 = !{i32 16, !21, !19}
!23 = !{i32 16, !24, !17}
!24 = !{!"explicit_arg_num", i32 5}
!25 = !{i32 16, !24, !19}
!26 = !{i32 16, !27, !17}
!27 = !{!"explicit_arg_num", i32 6}
!28 = !{i32 16, !27, !19}
!29 = !{i32 16, !30, !17}
!30 = !{!"explicit_arg_num", i32 8}
!31 = !{i32 16, !30, !19}
!32 = !{i32 16, !33, !17}
!33 = !{!"explicit_arg_num", i32 9}
!34 = !{i32 16, !33, !19}
!35 = !{i32 14, !36}
!36 = !{!"explicit_arg_num", i32 0}
!37 = !{i32 14, !38}
!38 = !{!"explicit_arg_num", i32 4}
!39 = !{i32 14, !40}
!40 = !{!"explicit_arg_num", i32 7}
!41 = !{!"sub_group_size", i32 8}
!42 = !{!"ModuleMD", !43, !44, !141, !308, !339, !340, !344, !347, !348, !349, !384, !410, !423, !424, !425, !440, !441, !442, !443, !444, !445, !446, !447, !448, !449, !453, !454, !461, !462, !463, !464, !465, !466, !467, !468, !469, !470, !471, !472, !474, !478, !479, !480, !481, !482, !483, !484, !485, !486, !487, !488, !489, !490, !491, !492, !493, !494, !232, !495, !496, !497, !499, !502, !503, !504, !506, !507, !508}
!43 = !{!"isPrecise", i1 false}
!44 = !{!"compOpt", !45, !46, !47, !48, !49, !50, !51, !52, !53, !54, !55, !56, !57, !58, !59, !60, !61, !62, !63, !64, !65, !66, !67, !68, !69, !70, !71, !72, !73, !74, !75, !76, !77, !78, !79, !80, !81, !82, !83, !84, !85, !86, !87, !88, !89, !90, !91, !92, !93, !94, !95, !96, !97, !98, !99, !100, !101, !102, !103, !104, !105, !106, !107, !108, !109, !110, !111, !112, !113, !114, !115, !116, !117, !118, !119, !120, !121, !122, !123, !124, !125, !126, !127, !128, !129, !130, !131, !132, !133, !134, !135, !136, !137, !138, !139, !140}
!45 = !{!"DenormsAreZero", i1 false}
!46 = !{!"BFTFDenormsAreZero", i1 false}
!47 = !{!"CorrectlyRoundedDivSqrt", i1 false}
!48 = !{!"OptDisable", i1 false}
!49 = !{!"MadEnable", i1 true}
!50 = !{!"NoSignedZeros", i1 false}
!51 = !{!"NoNaNs", i1 false}
!52 = !{!"FloatRoundingMode", i32 0}
!53 = !{!"FloatCvtIntRoundingMode", i32 3}
!54 = !{!"LoadCacheDefault", i32 4}
!55 = !{!"StoreCacheDefault", i32 7}
!56 = !{!"VISAPreSchedRPThreshold", i32 0}
!57 = !{!"SetLoopUnrollThreshold", i32 0}
!58 = !{!"UnsafeMathOptimizations", i1 false}
!59 = !{!"disableCustomUnsafeOpts", i1 false}
!60 = !{!"disableReducePow", i1 false}
!61 = !{!"disableSqrtOpt", i1 false}
!62 = !{!"FiniteMathOnly", i1 false}
!63 = !{!"FastRelaxedMath", i1 false}
!64 = !{!"DashGSpecified", i1 false}
!65 = !{!"FastCompilation", i1 false}
!66 = !{!"UseScratchSpacePrivateMemory", i1 true}
!67 = !{!"RelaxedBuiltins", i1 false}
!68 = !{!"SubgroupIndependentForwardProgressRequired", i1 true}
!69 = !{!"GreaterThan2GBBufferRequired", i1 true}
!70 = !{!"GreaterThan4GBBufferRequired", i1 false}
!71 = !{!"DisableA64WA", i1 false}
!72 = !{!"ForceEnableA64WA", i1 false}
!73 = !{!"PushConstantsEnable", i1 true}
!74 = !{!"HasPositivePointerOffset", i1 false}
!75 = !{!"HasBufferOffsetArg", i1 true}
!76 = !{!"BufferOffsetArgOptional", i1 true}
!77 = !{!"replaceGlobalOffsetsByZero", i1 false}
!78 = !{!"forcePixelShaderSIMDMode", i32 0}
!79 = !{!"ForceGeomFFShaderSIMDMode", i32 0}
!80 = !{!"pixelShaderDoNotAbortOnSpill", i1 false}
!81 = !{!"UniformWGS", i1 false}
!82 = !{!"disableVertexComponentPacking", i1 false}
!83 = !{!"disablePartialVertexComponentPacking", i1 false}
!84 = !{!"PreferBindlessImages", i1 false}
!85 = !{!"UseBindlessMode", i1 false}
!86 = !{!"UseLegacyBindlessMode", i1 true}
!87 = !{!"disableMathRefactoring", i1 false}
!88 = !{!"atomicBranch", i1 false}
!89 = !{!"spillCompression", i1 false}
!90 = !{!"DisableEarlyOut", i1 false}
!91 = !{!"ForceInt32DivRemEmu", i1 false}
!92 = !{!"ForceInt32DivRemEmuSP", i1 false}
!93 = !{!"WaveIntrinsicUsed", i1 false}
!94 = !{!"DisableMultiPolyPS", i1 false}
!95 = !{!"NeedTexture3DLODWA", i1 false}
!96 = !{!"DisableFastestSingleCSSIMD", i1 false}
!97 = !{!"DisableFastestLinearScan", i1 false}
!98 = !{!"UseStatelessforPrivateMemory", i1 false}
!99 = !{!"EnableTakeGlobalAddress", i1 false}
!100 = !{!"IsLibraryCompilation", i1 false}
!101 = !{!"LibraryCompileSIMDSize", i32 0}
!102 = !{!"FastVISACompile", i1 false}
!103 = !{!"MatchSinCosPi", i1 false}
!104 = !{!"ExcludeIRFromZEBinary", i1 false}
!105 = !{!"EmitZeBinVISASections", i1 false}
!106 = !{!"FP64GenEmulationEnabled", i1 false}
!107 = !{!"FP64GenConvEmulationEnabled", i1 false}
!108 = !{!"allowDisableRematforCS", i1 false}
!109 = !{!"DisableIncSpillCostAllAddrTaken", i1 false}
!110 = !{!"DisableCPSOmaskWA", i1 false}
!111 = !{!"DisableFastestGopt", i1 false}
!112 = !{!"WaForceHalfPromotionComputeShader", i1 false}
!113 = !{!"WaForceHalfPromotionPixelVertexShader", i1 false}
!114 = !{!"DisableConstantCoalescing", i1 false}
!115 = !{!"EnableUndefAlphaOutputAsRed", i1 true}
!116 = !{!"WaEnableALTModeVisaWA", i1 false}
!117 = !{!"WaEnableAtomicWaveFusion", i1 false}
!118 = !{!"WaEnableAtomicWaveFusionNonNullResource", i1 false}
!119 = !{!"WaEnableAtomicWaveFusionStateless", i1 false}
!120 = !{!"WaEnableAtomicWaveFusionTyped", i1 false}
!121 = !{!"ForceCBThroughSampler3D", i1 false}
!122 = !{!"WaStoreRawVectorToTypedWrite", i1 false}
!123 = !{!"WaLoadRawVectorToTypedRead", i1 false}
!124 = !{!"WaZeroSLMBeforeUse", i1 false}
!125 = !{!"WaFlagGroupTypedUAVGloballyCoherent", i1 false}
!126 = !{!"NewSpillCostFunction", i1 false}
!127 = !{!"EnableVRT", i1 false}
!128 = !{!"ForceLargeGRFNum4RQ", i1 false}
!129 = !{!"Enable2xGRFRetry", i1 false}
!130 = !{!"Detect2xGRFCandidate", i1 false}
!131 = !{!"EnableURBWritesMerging", i1 true}
!132 = !{!"DisableEUFusion", i1 false}
!133 = !{!"DisableFDivToFMulInvOpt", i1 false}
!134 = !{!"initializePhiSampleSourceWA", i1 false}
!135 = !{!"WaDisableSubspanUseNoMaskForCB", i1 false}
!136 = !{!"DisableLoosenSimd32Occu", i1 false}
!137 = !{!"FastestS1Options", i32 0}
!138 = !{!"DisableFastestForWaveIntrinsicsCS", i1 false}
!139 = !{!"ForceLinearWalkOnLinearUAV", i1 false}
!140 = !{!"DisableLscSamplerRouting", i1 false}
!141 = !{!"FuncMD", !142, !143}
!142 = !{!"FuncMDMap[0]", void (float addrspace(1)*, %"class.sycl::_V1::range"*, %"class.sycl::_V1::range"*, i64, %"class.sycl::_V1::ext::oneapi::bfloat16" addrspace(1)*, %"class.sycl::_V1::range"*, %"class.sycl::_V1::range"*, %"class.sycl::_V1::ext::oneapi::bfloat16" addrspace(1)*, %"class.sycl::_V1::range"*, %"class.sycl::_V1::range"*, <8 x i32>, <8 x i32>, <3 x i32>, <3 x i32>, i16, i16, i16, i8*, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i32, i32, i32)* @_ZTS7imatrixIfLm32ELm32ELm16EE}
!143 = !{!"FuncMDValue[0]", !144, !145, !149, !150, !151, !152, !153, !154, !155, !177, !224, !225, !226, !227, !228, !229, !230, !231, !232, !233, !234, !235, !236, !237, !238, !239, !240, !251, !262, !273, !284, !295, !306, !307}
!144 = !{!"localOffsets"}
!145 = !{!"workGroupWalkOrder", !146, !147, !148}
!146 = !{!"dim0", i32 0}
!147 = !{!"dim1", i32 1}
!148 = !{!"dim2", i32 2}
!149 = !{!"funcArgs"}
!150 = !{!"functionType", !"KernelFunction"}
!151 = !{!"inlineDynConstants"}
!152 = !{!"inlineDynRootConstant"}
!153 = !{!"inlineDynConstantDescTable"}
!154 = !{!"m_pInterestingConstants"}
!155 = !{!"rtInfo", !156, !157, !158, !159, !160, !161, !162, !163, !164, !165, !166, !167, !168, !169, !170, !171, !175, !176, !127}
!156 = !{!"callableShaderType", !"NumberOfCallableShaderTypes"}
!157 = !{!"isContinuation", i1 false}
!158 = !{!"hasTraceRayPayload", i1 false}
!159 = !{!"hasHitAttributes", i1 false}
!160 = !{!"hasCallableData", i1 false}
!161 = !{!"ShaderStackSize", i32 0}
!162 = !{!"ShaderHash", i64 0}
!163 = !{!"ShaderName", !""}
!164 = !{!"ParentName", !""}
!165 = !{!"SlotNum", i1* null}
!166 = !{!"NOSSize", i32 0}
!167 = !{!"globalRootSignatureSize", i32 0}
!168 = !{!"Entries"}
!169 = !{!"SpillUnions"}
!170 = !{!"CustomHitAttrSizeInBytes", i32 0}
!171 = !{!"Types", !172, !173, !174}
!172 = !{!"FrameStartTys"}
!173 = !{!"ArgumentTys"}
!174 = !{!"FullFrameTys"}
!175 = !{!"Aliases"}
!176 = !{!"NumGRF", i32 0}
!177 = !{!"resAllocMD", !178, !179, !180, !181, !223}
!178 = !{!"uavsNumType", i32 4}
!179 = !{!"srvsNumType", i32 0}
!180 = !{!"samplersNumType", i32 0}
!181 = !{!"argAllocMDList", !182, !186, !189, !190, !191, !193, !194, !195, !197, !198, !199, !200, !201, !202, !203, !204, !205, !206, !208, !209, !210, !211, !212, !213, !214, !215, !216, !217, !218, !219, !220, !221, !222}
!182 = !{!"argAllocMDListVec[0]", !183, !184, !185}
!183 = !{!"type", i32 1}
!184 = !{!"extensionType", i32 -1}
!185 = !{!"indexType", i32 0}
!186 = !{!"argAllocMDListVec[1]", !187, !184, !188}
!187 = !{!"type", i32 0}
!188 = !{!"indexType", i32 -1}
!189 = !{!"argAllocMDListVec[2]", !187, !184, !188}
!190 = !{!"argAllocMDListVec[3]", !187, !184, !188}
!191 = !{!"argAllocMDListVec[4]", !183, !184, !192}
!192 = !{!"indexType", i32 1}
!193 = !{!"argAllocMDListVec[5]", !187, !184, !188}
!194 = !{!"argAllocMDListVec[6]", !187, !184, !188}
!195 = !{!"argAllocMDListVec[7]", !183, !184, !196}
!196 = !{!"indexType", i32 2}
!197 = !{!"argAllocMDListVec[8]", !187, !184, !188}
!198 = !{!"argAllocMDListVec[9]", !187, !184, !188}
!199 = !{!"argAllocMDListVec[10]", !187, !184, !188}
!200 = !{!"argAllocMDListVec[11]", !187, !184, !188}
!201 = !{!"argAllocMDListVec[12]", !187, !184, !188}
!202 = !{!"argAllocMDListVec[13]", !187, !184, !188}
!203 = !{!"argAllocMDListVec[14]", !187, !184, !188}
!204 = !{!"argAllocMDListVec[15]", !187, !184, !188}
!205 = !{!"argAllocMDListVec[16]", !187, !184, !188}
!206 = !{!"argAllocMDListVec[17]", !183, !184, !207}
!207 = !{!"indexType", i32 3}
!208 = !{!"argAllocMDListVec[18]", !187, !184, !188}
!209 = !{!"argAllocMDListVec[19]", !187, !184, !188}
!210 = !{!"argAllocMDListVec[20]", !187, !184, !188}
!211 = !{!"argAllocMDListVec[21]", !187, !184, !188}
!212 = !{!"argAllocMDListVec[22]", !187, !184, !188}
!213 = !{!"argAllocMDListVec[23]", !187, !184, !188}
!214 = !{!"argAllocMDListVec[24]", !187, !184, !188}
!215 = !{!"argAllocMDListVec[25]", !187, !184, !188}
!216 = !{!"argAllocMDListVec[26]", !187, !184, !188}
!217 = !{!"argAllocMDListVec[27]", !187, !184, !188}
!218 = !{!"argAllocMDListVec[28]", !187, !184, !188}
!219 = !{!"argAllocMDListVec[29]", !187, !184, !188}
!220 = !{!"argAllocMDListVec[30]", !187, !184, !188}
!221 = !{!"argAllocMDListVec[31]", !187, !184, !188}
!222 = !{!"argAllocMDListVec[32]", !187, !184, !188}
!223 = !{!"inlineSamplersMD"}
!224 = !{!"maxByteOffsets"}
!225 = !{!"IsInitializer", i1 false}
!226 = !{!"IsFinalizer", i1 false}
!227 = !{!"CompiledSubGroupsNumber", i32 0}
!228 = !{!"hasInlineVmeSamplers", i1 false}
!229 = !{!"localSize", i32 0}
!230 = !{!"localIDPresent", i1 false}
!231 = !{!"groupIDPresent", i1 false}
!232 = !{!"privateMemoryPerWI", i32 0}
!233 = !{!"prevFPOffset", i32 0}
!234 = !{!"globalIDPresent", i1 false}
!235 = !{!"hasSyncRTCalls", i1 false}
!236 = !{!"hasNonKernelArgLoad", i1 false}
!237 = !{!"hasNonKernelArgStore", i1 false}
!238 = !{!"hasNonKernelArgAtomic", i1 false}
!239 = !{!"UserAnnotations"}
!240 = !{!"m_OpenCLArgAddressSpaces", !241, !242, !243, !244, !245, !246, !247, !248, !249, !250}
!241 = !{!"m_OpenCLArgAddressSpacesVec[0]", i32 1}
!242 = !{!"m_OpenCLArgAddressSpacesVec[1]", i32 0}
!243 = !{!"m_OpenCLArgAddressSpacesVec[2]", i32 0}
!244 = !{!"m_OpenCLArgAddressSpacesVec[3]", i32 0}
!245 = !{!"m_OpenCLArgAddressSpacesVec[4]", i32 1}
!246 = !{!"m_OpenCLArgAddressSpacesVec[5]", i32 0}
!247 = !{!"m_OpenCLArgAddressSpacesVec[6]", i32 0}
!248 = !{!"m_OpenCLArgAddressSpacesVec[7]", i32 1}
!249 = !{!"m_OpenCLArgAddressSpacesVec[8]", i32 0}
!250 = !{!"m_OpenCLArgAddressSpacesVec[9]", i32 0}
!251 = !{!"m_OpenCLArgAccessQualifiers", !252, !253, !254, !255, !256, !257, !258, !259, !260, !261}
!252 = !{!"m_OpenCLArgAccessQualifiersVec[0]", !"none"}
!253 = !{!"m_OpenCLArgAccessQualifiersVec[1]", !"none"}
!254 = !{!"m_OpenCLArgAccessQualifiersVec[2]", !"none"}
!255 = !{!"m_OpenCLArgAccessQualifiersVec[3]", !"none"}
!256 = !{!"m_OpenCLArgAccessQualifiersVec[4]", !"none"}
!257 = !{!"m_OpenCLArgAccessQualifiersVec[5]", !"none"}
!258 = !{!"m_OpenCLArgAccessQualifiersVec[6]", !"none"}
!259 = !{!"m_OpenCLArgAccessQualifiersVec[7]", !"none"}
!260 = !{!"m_OpenCLArgAccessQualifiersVec[8]", !"none"}
!261 = !{!"m_OpenCLArgAccessQualifiersVec[9]", !"none"}
!262 = !{!"m_OpenCLArgTypes", !263, !264, !265, !266, !267, !268, !269, !270, !271, !272}
!263 = !{!"m_OpenCLArgTypesVec[0]", !"float*"}
!264 = !{!"m_OpenCLArgTypesVec[1]", !"class.sycl::_V1::range"}
!265 = !{!"m_OpenCLArgTypesVec[2]", !"class.sycl::_V1::range"}
!266 = !{!"m_OpenCLArgTypesVec[3]", !"long"}
!267 = !{!"m_OpenCLArgTypesVec[4]", !"class.sycl::_V1::ext::oneapi::bfloat16*"}
!268 = !{!"m_OpenCLArgTypesVec[5]", !"class.sycl::_V1::range"}
!269 = !{!"m_OpenCLArgTypesVec[6]", !"class.sycl::_V1::range"}
!270 = !{!"m_OpenCLArgTypesVec[7]", !"class.sycl::_V1::ext::oneapi::bfloat16*"}
!271 = !{!"m_OpenCLArgTypesVec[8]", !"class.sycl::_V1::range"}
!272 = !{!"m_OpenCLArgTypesVec[9]", !"class.sycl::_V1::range"}
!273 = !{!"m_OpenCLArgBaseTypes", !274, !275, !276, !277, !278, !279, !280, !281, !282, !283}
!274 = !{!"m_OpenCLArgBaseTypesVec[0]", !"float*"}
!275 = !{!"m_OpenCLArgBaseTypesVec[1]", !"class.sycl::_V1::range"}
!276 = !{!"m_OpenCLArgBaseTypesVec[2]", !"class.sycl::_V1::range"}
!277 = !{!"m_OpenCLArgBaseTypesVec[3]", !"long"}
!278 = !{!"m_OpenCLArgBaseTypesVec[4]", !"class.sycl::_V1::ext::oneapi::bfloat16*"}
!279 = !{!"m_OpenCLArgBaseTypesVec[5]", !"class.sycl::_V1::range"}
!280 = !{!"m_OpenCLArgBaseTypesVec[6]", !"class.sycl::_V1::range"}
!281 = !{!"m_OpenCLArgBaseTypesVec[7]", !"class.sycl::_V1::ext::oneapi::bfloat16*"}
!282 = !{!"m_OpenCLArgBaseTypesVec[8]", !"class.sycl::_V1::range"}
!283 = !{!"m_OpenCLArgBaseTypesVec[9]", !"class.sycl::_V1::range"}
!284 = !{!"m_OpenCLArgTypeQualifiers", !285, !286, !287, !288, !289, !290, !291, !292, !293, !294}
!285 = !{!"m_OpenCLArgTypeQualifiersVec[0]", !""}
!286 = !{!"m_OpenCLArgTypeQualifiersVec[1]", !""}
!287 = !{!"m_OpenCLArgTypeQualifiersVec[2]", !""}
!288 = !{!"m_OpenCLArgTypeQualifiersVec[3]", !""}
!289 = !{!"m_OpenCLArgTypeQualifiersVec[4]", !""}
!290 = !{!"m_OpenCLArgTypeQualifiersVec[5]", !""}
!291 = !{!"m_OpenCLArgTypeQualifiersVec[6]", !""}
!292 = !{!"m_OpenCLArgTypeQualifiersVec[7]", !""}
!293 = !{!"m_OpenCLArgTypeQualifiersVec[8]", !""}
!294 = !{!"m_OpenCLArgTypeQualifiersVec[9]", !""}
!295 = !{!"m_OpenCLArgNames", !296, !297, !298, !299, !300, !301, !302, !303, !304, !305}
!296 = !{!"m_OpenCLArgNamesVec[0]", !""}
!297 = !{!"m_OpenCLArgNamesVec[1]", !""}
!298 = !{!"m_OpenCLArgNamesVec[2]", !""}
!299 = !{!"m_OpenCLArgNamesVec[3]", !""}
!300 = !{!"m_OpenCLArgNamesVec[4]", !""}
!301 = !{!"m_OpenCLArgNamesVec[5]", !""}
!302 = !{!"m_OpenCLArgNamesVec[6]", !""}
!303 = !{!"m_OpenCLArgNamesVec[7]", !""}
!304 = !{!"m_OpenCLArgNamesVec[8]", !""}
!305 = !{!"m_OpenCLArgNamesVec[9]", !""}
!306 = !{!"m_OpenCLArgScalarAsPointers"}
!307 = !{!"m_OptsToDisablePerFunc"}
!308 = !{!"pushInfo", !309, !310, !311, !315, !316, !317, !318, !319, !320, !321, !322, !335, !336, !337, !338}
!309 = !{!"pushableAddresses"}
!310 = !{!"bindlessPushInfo"}
!311 = !{!"dynamicBufferInfo", !312, !313, !314}
!312 = !{!"firstIndex", i32 0}
!313 = !{!"numOffsets", i32 0}
!314 = !{!"forceDisabled", i1 false}
!315 = !{!"MaxNumberOfPushedBuffers", i32 0}
!316 = !{!"inlineConstantBufferSlot", i32 -1}
!317 = !{!"inlineConstantBufferOffset", i32 -1}
!318 = !{!"inlineConstantBufferGRFOffset", i32 -1}
!319 = !{!"constants"}
!320 = !{!"inputs"}
!321 = !{!"constantReg"}
!322 = !{!"simplePushInfoArr", !323, !332, !333, !334}
!323 = !{!"simplePushInfoArrVec[0]", !324, !325, !326, !327, !328, !329, !330, !331}
!324 = !{!"cbIdx", i32 0}
!325 = !{!"pushableAddressGrfOffset", i32 -1}
!326 = !{!"pushableOffsetGrfOffset", i32 -1}
!327 = !{!"offset", i32 0}
!328 = !{!"size", i32 0}
!329 = !{!"isStateless", i1 false}
!330 = !{!"isBindless", i1 false}
!331 = !{!"simplePushLoads"}
!332 = !{!"simplePushInfoArrVec[1]", !324, !325, !326, !327, !328, !329, !330, !331}
!333 = !{!"simplePushInfoArrVec[2]", !324, !325, !326, !327, !328, !329, !330, !331}
!334 = !{!"simplePushInfoArrVec[3]", !324, !325, !326, !327, !328, !329, !330, !331}
!335 = !{!"simplePushBufferUsed", i32 0}
!336 = !{!"pushAnalysisWIInfos"}
!337 = !{!"inlineRTGlobalPtrOffset", i32 0}
!338 = !{!"rtSyncSurfPtrOffset", i32 0}
!339 = !{!"WaEnableICBPromotion", i1 false}
!340 = !{!"vsInfo", !341, !342, !343}
!341 = !{!"DrawIndirectBufferIndex", i32 -1}
!342 = !{!"vertexReordering", i32 -1}
!343 = !{!"MaxNumOfOutputs", i32 0}
!344 = !{!"hsInfo", !345, !346}
!345 = !{!"numPatchAttributesPatchBaseName", !""}
!346 = !{!"numVertexAttributesPatchBaseName", !""}
!347 = !{!"dsInfo", !343}
!348 = !{!"gsInfo", !343}
!349 = !{!"psInfo", !350, !351, !352, !353, !354, !355, !356, !357, !358, !359, !360, !361, !362, !363, !364, !365, !366, !367, !368, !369, !370, !371, !372, !373, !374, !375, !376, !377, !378, !379, !380, !381, !382, !383}
!350 = !{!"BlendStateDisabledMask", i8 0}
!351 = !{!"SkipSrc0Alpha", i1 false}
!352 = !{!"DualSourceBlendingDisabled", i1 false}
!353 = !{!"ForceEnableSimd32", i1 false}
!354 = !{!"outputDepth", i1 false}
!355 = !{!"outputStencil", i1 false}
!356 = !{!"outputMask", i1 false}
!357 = !{!"blendToFillEnabled", i1 false}
!358 = !{!"forceEarlyZ", i1 false}
!359 = !{!"hasVersionedLoop", i1 false}
!360 = !{!"forceSingleSourceRTWAfterDualSourceRTW", i1 false}
!361 = !{!"requestCPSizeRelevant", i1 false}
!362 = !{!"requestCPSize", i1 false}
!363 = !{!"texelMaskFastClearMode", !"Disabled"}
!364 = !{!"NumSamples", i8 0}
!365 = !{!"blendOptimizationMode"}
!366 = !{!"colorOutputMask"}
!367 = !{!"ProvokingVertexModeNosIndex", i32 0}
!368 = !{!"ProvokingVertexModeNosPatch", !""}
!369 = !{!"ProvokingVertexModeLast", !"Negative"}
!370 = !{!"VertexAttributesBypass", i1 false}
!371 = !{!"LegacyBaryAssignmentDisableLinear", i1 false}
!372 = !{!"LegacyBaryAssignmentDisableLinearNoPerspective", i1 false}
!373 = !{!"LegacyBaryAssignmentDisableLinearCentroid", i1 false}
!374 = !{!"LegacyBaryAssignmentDisableLinearNoPerspectiveCentroid", i1 false}
!375 = !{!"LegacyBaryAssignmentDisableLinearSample", i1 false}
!376 = !{!"LegacyBaryAssignmentDisableLinearNoPerspectiveSample", i1 false}
!377 = !{!"MeshShaderWAPerPrimitiveUserDataEnable", !"Negative"}
!378 = !{!"meshShaderWAPerPrimitiveUserDataEnablePatchName", !""}
!379 = !{!"generatePatchesForRTWriteSends", i1 false}
!380 = !{!"forceVMask", i1 false}
!381 = !{!"WaDisableVRS", i1 false}
!382 = !{!"RelaxMemoryVisibilityFromPSOrdering", i1 false}
!383 = !{!"WaEnableVMaskUnderNonUnifromCF", i1 false}
!384 = !{!"csInfo", !385, !386, !387, !388, !389, !56, !57, !390, !391, !392, !393, !394, !395, !396, !397, !398, !399, !400, !401, !402, !89, !403, !404, !405, !406, !407, !408, !409}
!385 = !{!"maxWorkGroupSize", i32 0}
!386 = !{!"waveSize", i32 0}
!387 = !{!"ComputeShaderSecondCompile"}
!388 = !{!"forcedSIMDSize", i8 0}
!389 = !{!"forceTotalGRFNum", i32 0}
!390 = !{!"forceSpillCompression", i1 false}
!391 = !{!"allowLowerSimd", i1 false}
!392 = !{!"disableSimd32Slicing", i1 false}
!393 = !{!"disableSplitOnSpill", i1 false}
!394 = !{!"enableNewSpillCostFunction", i1 false}
!395 = !{!"forceVISAPreSched", i1 false}
!396 = !{!"forceUniformBuffer", i1 false}
!397 = !{!"forceUniformSurfaceSampler", i1 false}
!398 = !{!"disableLocalIdOrderOptimizations", i1 false}
!399 = !{!"disableDispatchAlongY", i1 false}
!400 = !{!"neededThreadIdLayout", i1* null}
!401 = !{!"forceTileYWalk", i1 false}
!402 = !{!"atomicBranch", i32 0}
!403 = !{!"disableEarlyOut", i1 false}
!404 = !{!"walkOrderEnabled", i1 false}
!405 = !{!"walkOrderOverride", i32 0}
!406 = !{!"ResForHfPacking"}
!407 = !{!"hasWaveMatrix", i1 false}
!408 = !{!"constantFoldSimdSize", i1 false}
!409 = !{!"isNodeShader", i1 false}
!410 = !{!"msInfo", !411, !412, !413, !414, !415, !416, !417, !418, !419, !420, !421, !369, !367, !422}
!411 = !{!"PrimitiveTopology", i32 3}
!412 = !{!"MaxNumOfPrimitives", i32 0}
!413 = !{!"MaxNumOfVertices", i32 0}
!414 = !{!"MaxNumOfPerPrimitiveOutputs", i32 0}
!415 = !{!"MaxNumOfPerVertexOutputs", i32 0}
!416 = !{!"WorkGroupSize", i32 0}
!417 = !{!"WorkGroupMemorySizeInBytes", i32 0}
!418 = !{!"IndexFormat", i32 6}
!419 = !{!"SubgroupSize", i32 0}
!420 = !{!"VPandRTAIndexAutostripEnable", i1 false}
!421 = !{!"MeshShaderWAPerPrimitiveUserDataEnable", i1 false}
!422 = !{!"numPrimitiveAttributesPatchBaseName", !""}
!423 = !{!"taskInfo", !343, !416, !417, !419}
!424 = !{!"NBarrierCnt", i32 0}
!425 = !{!"rtInfo", !426, !427, !428, !429, !430, !431, !432, !433, !434, !435, !436, !437, !438, !439}
!426 = !{!"RayQueryAllocSizeInBytes", i32 0}
!427 = !{!"NumContinuations", i32 0}
!428 = !{!"RTAsyncStackAddrspace", i32 -1}
!429 = !{!"RTAsyncStackSurfaceStateOffset", i1* null}
!430 = !{!"SWHotZoneAddrspace", i32 -1}
!431 = !{!"SWHotZoneSurfaceStateOffset", i1* null}
!432 = !{!"SWStackAddrspace", i32 -1}
!433 = !{!"SWStackSurfaceStateOffset", i1* null}
!434 = !{!"RTSyncStackAddrspace", i32 -1}
!435 = !{!"RTSyncStackSurfaceStateOffset", i1* null}
!436 = !{!"doSyncDispatchRays", i1 false}
!437 = !{!"MemStyle", !"Xe"}
!438 = !{!"GlobalDataStyle", !"Xe"}
!439 = !{!"NeedsBTD", i1 true}
!440 = !{!"EnableTextureIndirection", i1 false}
!441 = !{!"EnableSamplerIndirection", i1 false}
!442 = !{!"samplerStateStride", i32 0}
!443 = !{!"samplerStateOffset", i32 0}
!444 = !{!"textureStateStride", i32 0}
!445 = !{!"textureStateOffset", i32 0}
!446 = !{!"CurUniqueIndirectIdx", i32 0}
!447 = !{!"inlineDynTextures"}
!448 = !{!"inlineResInfoData"}
!449 = !{!"immConstant", !450, !451, !452}
!450 = !{!"data"}
!451 = !{!"sizes"}
!452 = !{!"zeroIdxs"}
!453 = !{!"stringConstants"}
!454 = !{!"inlineBuffers", !455, !459, !460}
!455 = !{!"inlineBuffersVec[0]", !456, !457, !458}
!456 = !{!"alignment", i32 0}
!457 = !{!"allocSize", i64 0}
!458 = !{!"Buffer"}
!459 = !{!"inlineBuffersVec[1]", !456, !457, !458}
!460 = !{!"inlineBuffersVec[2]", !456, !457, !458}
!461 = !{!"GlobalPointerProgramBinaryInfos"}
!462 = !{!"ConstantPointerProgramBinaryInfos"}
!463 = !{!"GlobalBufferAddressRelocInfo"}
!464 = !{!"ConstantBufferAddressRelocInfo"}
!465 = !{!"forceLscCacheList"}
!466 = !{!"SrvMap"}
!467 = !{!"RootConstantBufferOffsetInBytes"}
!468 = !{!"RasterizerOrderedByteAddressBuffer"}
!469 = !{!"RasterizerOrderedViews"}
!470 = !{!"MinNOSPushConstantSize", i32 0}
!471 = !{!"inlineProgramScopeOffsets"}
!472 = !{!"shaderData", !473}
!473 = !{!"numReplicas", i32 0}
!474 = !{!"URBInfo", !475, !476, !477}
!475 = !{!"has64BVertexHeaderInput", i1 false}
!476 = !{!"has64BVertexHeaderOutput", i1 false}
!477 = !{!"hasVertexHeader", i1 true}
!478 = !{!"m_ForcePullModel", i1 false}
!479 = !{!"UseBindlessImage", i1 false}
!480 = !{!"enableRangeReduce", i1 false}
!481 = !{!"disableNewTrigFuncRangeReduction", i1 false}
!482 = !{!"enableFRemToSRemOpt", i1 false}
!483 = !{!"enableSampleptrToLdmsptrSample0", i1 false}
!484 = !{!"enableSampleLptrToLdmsptrSample0", i1 false}
!485 = !{!"WaForceSIMD32MicropolyRasterize", i1 false}
!486 = !{!"allowMatchMadOptimizationforVS", i1 false}
!487 = !{!"disableMatchMadOptimizationForCS", i1 false}
!488 = !{!"disableMemOptforNegativeOffsetLoads", i1 false}
!489 = !{!"enableThreeWayLoadSpiltOpt", i1 false}
!490 = !{!"statefulResourcesNotAliased", i1 false}
!491 = !{!"disableMixMode", i1 false}
!492 = !{!"genericAccessesResolved", i1 false}
!493 = !{!"disableSeparateSpillPvtScratchSpace", i1 false}
!494 = !{!"disableSeparateScratchWA", i1 false}
!495 = !{!"PrivateMemoryPerFG"}
!496 = !{!"m_OptsToDisable"}
!497 = !{!"capabilities", !498}
!498 = !{!"globalVariableDecorationsINTEL", i1 false}
!499 = !{!"m_ShaderResourceViewMcsMask", !500, !501}
!500 = !{!"m_ShaderResourceViewMcsMaskVec[0]", i64 0}
!501 = !{!"m_ShaderResourceViewMcsMaskVec[1]", i64 0}
!502 = !{!"computedDepthMode", i32 0}
!503 = !{!"isHDCFastClearShader", i1 false}
!504 = !{!"argRegisterReservations", !505}
!505 = !{!"argRegisterReservationsVec[0]", i32 0}
!506 = !{!"SIMD16_SpillThreshold", i8 0}
!507 = !{!"SIMD32_SpillThreshold", i8 0}
!508 = !{!"m_CacheControlOption", !509, !510, !511, !512}
!509 = !{!"LscLoadCacheControlOverride", i8 0}
!510 = !{!"LscStoreCacheControlOverride", i8 0}
!511 = !{!"TgmLoadCacheControlOverride", i8 0}
!512 = !{!"TgmStoreCacheControlOverride", i8 0}
!513 = !{i32 2, i32 0}
!514 = !{!"clang version 14.0.5"}
!515 = !{i32 1, !"wchar_size", i32 4}
!516 = !{!517}
!517 = !{i32 4469}
!518 = !{!519, !519, i64 0}
!519 = !{!"int", !520, i64 0}
!520 = !{!"omnipotent char", !521, i64 0}
!521 = !{!"Simple C/C++ TBAA"}
!522 = !{!517, !523}
!523 = !{i32 4470}
