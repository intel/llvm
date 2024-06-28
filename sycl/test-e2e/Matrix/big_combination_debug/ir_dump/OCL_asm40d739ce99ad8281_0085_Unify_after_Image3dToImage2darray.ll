; ------------------------------------------------
; OCL_asm40d739ce99ad8281_0085_Unify_after_Image3dToImage2darray.ll
; LLVM major version: 14
; ------------------------------------------------
target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v16:16:16-v24:32:32-v32:32:32-v48:64:64-v64:64:64-v96:128:128-v128:128:128-v192:256:256-v256:256:256-v512:512:512-v1024:1024:1024-n8:16:32"
target triple = "spir64-unknown-unknown"

%"class.sycl::_V1::ext::oneapi::bfloat16" = type { i16 }
%"class.sycl::_V1::range" = type { %"class.sycl::_V1::detail::array" }
%"class.sycl::_V1::detail::array" = type { [2 x i64] }

; Function Attrs: convergent nounwind
define spir_kernel void @_ZTS7imatrixIfLm32ELm32ELm16EE(float addrspace(1)* align 4 %0, %"class.sycl::_V1::range"* byval(%"class.sycl::_V1::range") align 8 %1, %"class.sycl::_V1::range"* byval(%"class.sycl::_V1::range") align 8 %2, i64 %3, %"class.sycl::_V1::ext::oneapi::bfloat16" addrspace(1)* align 2 %4, %"class.sycl::_V1::range"* byval(%"class.sycl::_V1::range") align 8 %5, %"class.sycl::_V1::range"* byval(%"class.sycl::_V1::range") align 8 %6, %"class.sycl::_V1::ext::oneapi::bfloat16" addrspace(1)* align 2 %7, %"class.sycl::_V1::range"* byval(%"class.sycl::_V1::range") align 8 %8, %"class.sycl::_V1::range"* byval(%"class.sycl::_V1::range") align 8 %9, <8 x i32> %r0, <8 x i32> %payloadHeader, <3 x i32> %localSize, <3 x i32> %enqueuedLocalSize, i16 %localIdX, i16 %localIdY, i16 %localIdZ, i8* %privateBase, i64 %const_reg_qword, i64 %const_reg_qword1, i64 %const_reg_qword2, i64 %const_reg_qword3, i64 %const_reg_qword4, i64 %const_reg_qword5, i64 %const_reg_qword6, i64 %const_reg_qword7, i64 %const_reg_qword8, i64 %const_reg_qword9, i64 %const_reg_qword10, i64 %const_reg_qword11, i32 %bufferOffset, i32 %bufferOffset12, i32 %bufferOffset13) #0 {
  %_alloca = alloca %"class.sycl::_V1::range", align 8
  %11 = bitcast %"class.sycl::_V1::range"* %_alloca to i8*
  %12 = getelementptr i8, i8* %11, i32 0
  %13 = bitcast i8* %12 to i64*
  store i64 %const_reg_qword, i64* %13, align 8
  %14 = getelementptr i8, i8* %11, i32 8
  %15 = bitcast i8* %14 to i64*
  store i64 %const_reg_qword1, i64* %15, align 8
  %_alloca73 = alloca %"class.sycl::_V1::range", align 8
  %16 = bitcast %"class.sycl::_V1::range"* %_alloca73 to i8*
  %17 = getelementptr i8, i8* %16, i32 0
  %18 = bitcast i8* %17 to i64*
  store i64 %const_reg_qword2, i64* %18, align 8
  %19 = getelementptr i8, i8* %16, i32 8
  %20 = bitcast i8* %19 to i64*
  store i64 %const_reg_qword3, i64* %20, align 8
  %_alloca74 = alloca %"class.sycl::_V1::range", align 8
  %21 = bitcast %"class.sycl::_V1::range"* %_alloca74 to i8*
  %22 = getelementptr i8, i8* %21, i32 0
  %23 = bitcast i8* %22 to i64*
  store i64 %const_reg_qword4, i64* %23, align 8
  %24 = getelementptr i8, i8* %21, i32 8
  %25 = bitcast i8* %24 to i64*
  store i64 %const_reg_qword5, i64* %25, align 8
  %_alloca75 = alloca %"class.sycl::_V1::range", align 8
  %26 = bitcast %"class.sycl::_V1::range"* %_alloca75 to i8*
  %27 = getelementptr i8, i8* %26, i32 0
  %28 = bitcast i8* %27 to i64*
  store i64 %const_reg_qword6, i64* %28, align 8
  %29 = getelementptr i8, i8* %26, i32 8
  %30 = bitcast i8* %29 to i64*
  store i64 %const_reg_qword7, i64* %30, align 8
  %_alloca76 = alloca %"class.sycl::_V1::range", align 8
  %31 = bitcast %"class.sycl::_V1::range"* %_alloca76 to i8*
  %32 = getelementptr i8, i8* %31, i32 0
  %33 = bitcast i8* %32 to i64*
  store i64 %const_reg_qword8, i64* %33, align 8
  %34 = getelementptr i8, i8* %31, i32 8
  %35 = bitcast i8* %34 to i64*
  store i64 %const_reg_qword9, i64* %35, align 8
  %_alloca77 = alloca %"class.sycl::_V1::range", align 8
  %36 = bitcast %"class.sycl::_V1::range"* %_alloca77 to i8*
  %37 = getelementptr i8, i8* %36, i32 0
  %38 = bitcast i8* %37 to i64*
  store i64 %const_reg_qword10, i64* %38, align 8
  %39 = getelementptr i8, i8* %36, i32 8
  %40 = bitcast i8* %39 to i64*
  store i64 %const_reg_qword11, i64* %40, align 8
  %41 = alloca <32 x i32>, align 128
  %42 = alloca <32 x i32>, align 128
  %43 = alloca [2 x <64 x float>], align 256
  %44 = alloca [2 x <64 x float>], align 256
  %45 = bitcast %"class.sycl::_V1::range"* %_alloca to i64*
  %46 = getelementptr inbounds i64, i64* %45, i64 1
  %47 = load i64, i64* %46, align 8
  %48 = bitcast %"class.sycl::_V1::range"* %_alloca73 to i64*
  %49 = load i64, i64* %48, align 8
  %50 = bitcast %"class.sycl::_V1::range"* %_alloca73 to i64*
  %51 = getelementptr inbounds i64, i64* %50, i64 1
  %52 = load i64, i64* %51, align 8
  %53 = mul i64 %49, %47
  %54 = getelementptr float, float addrspace(1)* %0, i64 %53
  %55 = getelementptr float, float addrspace(1)* %54, i64 %52
  %56 = bitcast %"class.sycl::_V1::range"* %_alloca74 to i64*
  %57 = getelementptr inbounds i64, i64* %56, i64 1
  %58 = load i64, i64* %57, align 8
  %59 = bitcast %"class.sycl::_V1::range"* %_alloca75 to i64*
  %60 = load i64, i64* %59, align 8
  %61 = bitcast %"class.sycl::_V1::range"* %_alloca75 to i64*
  %62 = getelementptr inbounds i64, i64* %61, i64 1
  %63 = load i64, i64* %62, align 8
  %64 = mul i64 %60, %58
  %65 = getelementptr %"class.sycl::_V1::ext::oneapi::bfloat16", %"class.sycl::_V1::ext::oneapi::bfloat16" addrspace(1)* %4, i64 %64
  %66 = getelementptr %"class.sycl::_V1::ext::oneapi::bfloat16", %"class.sycl::_V1::ext::oneapi::bfloat16" addrspace(1)* %65, i64 %63
  %67 = bitcast %"class.sycl::_V1::range"* %_alloca76 to i64*
  %68 = getelementptr inbounds i64, i64* %67, i64 1
  %69 = load i64, i64* %68, align 8
  %70 = bitcast %"class.sycl::_V1::range"* %_alloca77 to i64*
  %71 = load i64, i64* %70, align 8
  %72 = bitcast %"class.sycl::_V1::range"* %_alloca77 to i64*
  %73 = getelementptr inbounds i64, i64* %72, i64 1
  %74 = load i64, i64* %73, align 8
  %75 = mul i64 %71, %69
  %76 = getelementptr %"class.sycl::_V1::ext::oneapi::bfloat16", %"class.sycl::_V1::ext::oneapi::bfloat16" addrspace(1)* %7, i64 %75
  %77 = getelementptr %"class.sycl::_V1::ext::oneapi::bfloat16", %"class.sycl::_V1::ext::oneapi::bfloat16" addrspace(1)* %76, i64 %74
  %cmpDim = icmp eq i32 1, 0
  %tmpOffsetR0 = select i1 %cmpDim, i32 1, i32 5
  %offsetR0 = add i32 1, %tmpOffsetR0
  %groupId = extractelement <8 x i32> %r0, i32 %offsetR0
  %enqueuedLocalSize14 = extractelement <3 x i32> %enqueuedLocalSize, i32 1
  %78 = mul i32 %enqueuedLocalSize14, %groupId
  %localIdY15 = zext i16 %localIdY to i32
  %79 = icmp ult i32 %localIdY15, 65536
  call void @llvm.assume(i1 %79) #6
  %80 = add i32 %localIdY15, %78
  %globalOffset = extractelement <8 x i32> %payloadHeader, i32 1
  %81 = add i32 %80, %globalOffset
  %82 = zext i32 %81 to i64
  %83 = icmp ult i64 %82, 2147483648
  call void @llvm.assume(i1 %83)
  %cmpDim16 = icmp eq i32 0, 0
  %tmpOffsetR017 = select i1 %cmpDim16, i32 1, i32 5
  %offsetR018 = add i32 0, %tmpOffsetR017
  %groupId19 = extractelement <8 x i32> %r0, i32 %offsetR018
  %enqueuedLocalSize20 = extractelement <3 x i32> %enqueuedLocalSize, i32 0
  %84 = mul i32 %enqueuedLocalSize20, %groupId19
  %localIdX21 = zext i16 %localIdX to i32
  %85 = icmp ult i32 %localIdX21, 65536
  call void @llvm.assume(i1 %85) #6
  %86 = add i32 %localIdX21, %84
  %globalOffset22 = extractelement <8 x i32> %payloadHeader, i32 0
  %87 = add i32 %86, %globalOffset22
  %88 = zext i32 %87 to i64
  %89 = icmp ult i64 %88, 2147483648
  call void @llvm.assume(i1 %89)
  %localIdY23 = zext i16 %localIdY to i32
  %90 = icmp ult i32 %localIdY23, 65536
  call void @llvm.assume(i1 %90) #6
  %91 = zext i32 %localIdY23 to i64
  %92 = icmp ult i64 %91, 2147483648
  call void @llvm.assume(i1 %92)
  %93 = sub nsw i64 %82, %91, !spirv.Decorations !516
  %localIdX24 = zext i16 %localIdX to i32
  %94 = icmp ult i32 %localIdX24, 65536
  call void @llvm.assume(i1 %94) #6
  %95 = zext i32 %localIdX24 to i64
  %96 = icmp ult i64 %95, 2147483648
  call void @llvm.assume(i1 %96)
  %97 = sub nsw i64 %88, %95, !spirv.Decorations !516
  %98 = add i64 %53, %52
  %99 = sub i64 0, %98
  %100 = getelementptr inbounds float, float addrspace(1)* %55, i64 %99
  %101 = shl nsw i64 %93, 11, !spirv.Decorations !516
  %102 = getelementptr inbounds float, float addrspace(1)* %100, i64 %101
  %103 = udiv i64 %97, %3
  %104 = shl i64 %103, 5
  %105 = getelementptr inbounds float, float addrspace(1)* %102, i64 %104
  %106 = bitcast [2 x <64 x float>]* %44 to i8*
  %.ascast.i3 = bitcast float addrspace(1)* %105 to i8 addrspace(1)*
  %localSize25 = extractelement <3 x i32> %localSize, i32 0
  %localSize26 = extractelement <3 x i32> %localSize, i32 1
  %107 = mul i32 %localSize26, %localSize25
  %localSize27 = extractelement <3 x i32> %localSize, i32 2
  %108 = mul i32 %107, %localSize27
  %109 = add i32 8, -1
  %110 = and i32 %109, %108
  %111 = icmp eq i32 %110, 0
  br i1 %111, label %125, label %112

112:                                              ; preds = %10
  %localIdZ28 = zext i16 %localIdZ to i32
  %localSize29 = extractelement <3 x i32> %localSize, i32 1
  %113 = mul i32 %localSize29, %localIdZ28
  %localIdY30 = zext i16 %localIdY to i32
  %114 = add i32 %113, %localIdY30
  %localSize31 = extractelement <3 x i32> %localSize, i32 0
  %115 = mul i32 %114, %localSize31
  %localIdX32 = zext i16 %localIdX to i32
  %116 = add i32 %115, %localIdX32
  %117 = udiv i32 %116, 8
  %localSize33 = extractelement <3 x i32> %localSize, i32 0
  %localSize34 = extractelement <3 x i32> %localSize, i32 1
  %118 = mul i32 %localSize34, %localSize33
  %localSize35 = extractelement <3 x i32> %localSize, i32 2
  %119 = mul i32 %118, %localSize35
  %120 = add i32 %119, -1
  %121 = add i32 %120, 8
  %122 = udiv i32 %121, 8
  %123 = add i32 %122, -1
  %124 = icmp ult i32 %117, %123
  br i1 %124, label %125, label %_Z18get_sub_group_sizev.exit.i4

125:                                              ; preds = %112, %10
  br label %_Z18get_sub_group_sizev.exit.i4

_Z18get_sub_group_sizev.exit.i4:                  ; preds = %125, %112
  %126 = phi i32 [ 8, %125 ], [ %110, %112 ]
  %127 = bitcast i8 addrspace(1)* %.ascast.i3 to i32 addrspace(1)*
  %simdLaneId16 = call i16 @llvm.genx.GenISA.simdLaneId()
  %simdLaneId = zext i16 %simdLaneId16 to i32
  %128 = sdiv i32 %126, 32
  %129 = bitcast i8* %106 to i32*
  %130 = freeze i32 %simdLaneId
  %131 = sdiv i32 %130, 32
  %132 = mul i32 %131, 32
  %133 = sub i32 %130, %132
  %134 = sext i32 %133 to i64
  br label %135

135:                                              ; preds = %146, %_Z18get_sub_group_sizev.exit.i4
  %136 = phi i32 [ 0, %_Z18get_sub_group_sizev.exit.i4 ], [ %150, %146 ]
  %137 = mul nsw i32 %136, %128
  %138 = add nsw i32 %137, %131
  %139 = icmp slt i32 %138, 32
  br i1 %139, label %140, label %146

140:                                              ; preds = %135
  %141 = sext i32 %138 to i64
  %142 = mul nsw i64 %141, 64
  %143 = add nsw i64 %142, %134
  %144 = getelementptr inbounds i32, i32 addrspace(1)* %127, i64 %143
  %145 = load i32, i32 addrspace(1)* %144, align 4, !tbaa !518
  br label %146

146:                                              ; preds = %140, %135
  %147 = phi i32 [ %145, %140 ], [ 0, %135 ]
  %148 = zext i32 %136 to i64
  %149 = getelementptr inbounds i32, i32* %129, i64 %148
  store i32 %147, i32* %149, align 4, !tbaa !518
  %150 = add nuw nsw i32 %136, 1
  %151 = icmp ult i32 %136, 127
  br i1 %151, label %135, label %__builtin_spriv_OpJointMatrixLoadINTEL_Accumulator_RowMajor_32x32_i32_128_global_v8i8_pi32_i32.23.exit

__builtin_spriv_OpJointMatrixLoadINTEL_Accumulator_RowMajor_32x32_i32_128_global_v8i8_pi32_i32.23.exit: ; preds = %146
  %152 = bitcast [2 x <64 x float>]* %44 to <64 x float>*
  %153 = load <64 x float>, <64 x float>* %152, align 256
  %154 = getelementptr <64 x float>, <64 x float>* %152, i32 1
  %155 = load <64 x float>, <64 x float>* %154, align 256
  %156 = insertvalue [2 x <64 x float>] undef, <64 x float> %153, 0
  %157 = insertvalue [2 x <64 x float>] %156, <64 x float> %155, 1
  %158 = add i64 %64, %63
  %159 = sub i64 0, %158
  %160 = getelementptr inbounds %"class.sycl::_V1::ext::oneapi::bfloat16", %"class.sycl::_V1::ext::oneapi::bfloat16" addrspace(1)* %66, i64 %159
  %161 = shl nsw i64 %93, 10, !spirv.Decorations !516
  %162 = getelementptr inbounds %"class.sycl::_V1::ext::oneapi::bfloat16", %"class.sycl::_V1::ext::oneapi::bfloat16" addrspace(1)* %160, i64 %161
  %163 = add i64 %75, %74
  %164 = sub i64 0, %163
  %165 = getelementptr inbounds %"class.sycl::_V1::ext::oneapi::bfloat16", %"class.sycl::_V1::ext::oneapi::bfloat16" addrspace(1)* %77, i64 %164
  %166 = shl i64 %103, 6
  %167 = getelementptr %"class.sycl::_V1::ext::oneapi::bfloat16", %"class.sycl::_V1::ext::oneapi::bfloat16" addrspace(1)* %165, i64 %166
  br label %168

168:                                              ; preds = %__builtin_spriv_OpJointMatrixLoadINTEL_PackedB_PackedB_16x32_i16_32_global_v8i8_pi32_i32.25.exit, %__builtin_spriv_OpJointMatrixLoadINTEL_Accumulator_RowMajor_32x32_i32_128_global_v8i8_pi32_i32.23.exit
  %169 = phi i32 [ 0, %__builtin_spriv_OpJointMatrixLoadINTEL_Accumulator_RowMajor_32x32_i32_128_global_v8i8_pi32_i32.23.exit ], [ %254, %__builtin_spriv_OpJointMatrixLoadINTEL_PackedB_PackedB_16x32_i16_32_global_v8i8_pi32_i32.25.exit ]
  %170 = icmp ult i32 %169, 2
  br i1 %170, label %171, label %255

171:                                              ; preds = %168
  %172 = shl nuw nsw i32 %169, 4, !spirv.Decorations !522
  %173 = zext i32 %172 to i64
  %174 = getelementptr inbounds %"class.sycl::_V1::ext::oneapi::bfloat16", %"class.sycl::_V1::ext::oneapi::bfloat16" addrspace(1)* %162, i64 %173
  %175 = bitcast <32 x i32>* %42 to i8*
  %.ascast.i5 = bitcast %"class.sycl::_V1::ext::oneapi::bfloat16" addrspace(1)* %174 to i8 addrspace(1)*
  %localSize36 = extractelement <3 x i32> %localSize, i32 0
  %localSize37 = extractelement <3 x i32> %localSize, i32 1
  %176 = mul i32 %localSize37, %localSize36
  %localSize38 = extractelement <3 x i32> %localSize, i32 2
  %177 = mul i32 %176, %localSize38
  %178 = add i32 8, -1
  %179 = and i32 %178, %177
  %180 = icmp eq i32 %179, 0
  br i1 %180, label %194, label %181

181:                                              ; preds = %171
  %localIdZ39 = zext i16 %localIdZ to i32
  %localSize40 = extractelement <3 x i32> %localSize, i32 1
  %182 = mul i32 %localSize40, %localIdZ39
  %localIdY41 = zext i16 %localIdY to i32
  %183 = add i32 %182, %localIdY41
  %localSize42 = extractelement <3 x i32> %localSize, i32 0
  %184 = mul i32 %183, %localSize42
  %localIdX43 = zext i16 %localIdX to i32
  %185 = add i32 %184, %localIdX43
  %186 = udiv i32 %185, 8
  %localSize44 = extractelement <3 x i32> %localSize, i32 0
  %localSize45 = extractelement <3 x i32> %localSize, i32 1
  %187 = mul i32 %localSize45, %localSize44
  %localSize46 = extractelement <3 x i32> %localSize, i32 2
  %188 = mul i32 %187, %localSize46
  %189 = add i32 %188, -1
  %190 = add i32 %189, 8
  %191 = udiv i32 %190, 8
  %192 = add i32 %191, -1
  %193 = icmp ult i32 %186, %192
  br i1 %193, label %194, label %_Z18get_sub_group_sizev.exit.i6

194:                                              ; preds = %181, %171
  br label %_Z18get_sub_group_sizev.exit.i6

_Z18get_sub_group_sizev.exit.i6:                  ; preds = %194, %181
  %195 = bitcast i8* %175 to i32*
  %196 = bitcast i8 addrspace(1)* %.ascast.i5 to i32 addrspace(1)*
  br label %197

197:                                              ; preds = %197, %_Z18get_sub_group_sizev.exit.i6
  %198 = phi i32 [ 0, %_Z18get_sub_group_sizev.exit.i6 ], [ %204, %197 ]
  %199 = zext i32 %198 to i64
  %200 = mul nsw i64 16, %199
  %201 = getelementptr inbounds i32, i32 addrspace(1)* %196, i64 %200
  %202 = call i32 @llvm.genx.GenISA.simdBlockRead.i32.p1i32(i32 addrspace(1)* %201)
  %203 = getelementptr inbounds i32, i32* %195, i64 %199
  store i32 %202, i32* %203, align 4, !tbaa !518
  %204 = add nuw nsw i32 %198, 1
  %205 = icmp ult i32 %198, 31
  br i1 %205, label %197, label %__builtin_spriv_OpJointMatrixLoadINTEL_PackedA_RowMajor_32x16_i16_32_global_v8i8_pi32_i32.24.exit

__builtin_spriv_OpJointMatrixLoadINTEL_PackedA_RowMajor_32x16_i16_32_global_v8i8_pi32_i32.24.exit: ; preds = %197
  %206 = shl nuw nsw i64 %173, 6, !spirv.Decorations !522
  %207 = getelementptr %"class.sycl::_V1::ext::oneapi::bfloat16", %"class.sycl::_V1::ext::oneapi::bfloat16" addrspace(1)* %167, i64 %206
  %208 = bitcast <32 x i32>* %41 to i8*
  %.ascast.i7 = bitcast %"class.sycl::_V1::ext::oneapi::bfloat16" addrspace(1)* %207 to i8 addrspace(1)*
  %localSize47 = extractelement <3 x i32> %localSize, i32 0
  %localSize48 = extractelement <3 x i32> %localSize, i32 1
  %209 = mul i32 %localSize48, %localSize47
  %localSize49 = extractelement <3 x i32> %localSize, i32 2
  %210 = mul i32 %209, %localSize49
  %211 = add i32 8, -1
  %212 = and i32 %211, %210
  %213 = icmp eq i32 %212, 0
  br i1 %213, label %227, label %214

214:                                              ; preds = %__builtin_spriv_OpJointMatrixLoadINTEL_PackedA_RowMajor_32x16_i16_32_global_v8i8_pi32_i32.24.exit
  %localIdZ50 = zext i16 %localIdZ to i32
  %localSize51 = extractelement <3 x i32> %localSize, i32 1
  %215 = mul i32 %localSize51, %localIdZ50
  %localIdY52 = zext i16 %localIdY to i32
  %216 = add i32 %215, %localIdY52
  %localSize53 = extractelement <3 x i32> %localSize, i32 0
  %217 = mul i32 %216, %localSize53
  %localIdX54 = zext i16 %localIdX to i32
  %218 = add i32 %217, %localIdX54
  %219 = udiv i32 %218, 8
  %localSize55 = extractelement <3 x i32> %localSize, i32 0
  %localSize56 = extractelement <3 x i32> %localSize, i32 1
  %220 = mul i32 %localSize56, %localSize55
  %localSize57 = extractelement <3 x i32> %localSize, i32 2
  %221 = mul i32 %220, %localSize57
  %222 = add i32 %221, -1
  %223 = add i32 %222, 8
  %224 = udiv i32 %223, 8
  %225 = add i32 %224, -1
  %226 = icmp ult i32 %219, %225
  br i1 %226, label %227, label %_Z18get_sub_group_sizev.exit.i8

227:                                              ; preds = %214, %__builtin_spriv_OpJointMatrixLoadINTEL_PackedA_RowMajor_32x16_i16_32_global_v8i8_pi32_i32.24.exit
  br label %_Z18get_sub_group_sizev.exit.i8

_Z18get_sub_group_sizev.exit.i8:                  ; preds = %227, %214
  %228 = phi i32 [ 8, %227 ], [ %212, %214 ]
  %229 = bitcast i8 addrspace(1)* %.ascast.i7 to i32 addrspace(1)*
  %simdLaneId1669 = call i16 @llvm.genx.GenISA.simdLaneId()
  %simdLaneId70 = zext i16 %simdLaneId1669 to i32
  %230 = sdiv i32 %228, 32
  %231 = bitcast i8* %208 to i32*
  %232 = freeze i32 %simdLaneId70
  %233 = sdiv i32 %232, 32
  %234 = mul i32 %233, 32
  %235 = sub i32 %232, %234
  %236 = sext i32 %235 to i64
  br label %237

237:                                              ; preds = %248, %_Z18get_sub_group_sizev.exit.i8
  %238 = phi i32 [ 0, %_Z18get_sub_group_sizev.exit.i8 ], [ %252, %248 ]
  %239 = mul nsw i32 %238, %230
  %240 = add nsw i32 %239, %233
  %241 = icmp slt i32 %240, 8
  br i1 %241, label %242, label %248

242:                                              ; preds = %237
  %243 = sext i32 %240 to i64
  %244 = mul nsw i64 64, %243
  %245 = add nsw i64 %244, %236
  %246 = getelementptr inbounds i32, i32 addrspace(1)* %229, i64 %245
  %247 = load i32, i32 addrspace(1)* %246, align 4, !tbaa !518
  br label %248

248:                                              ; preds = %242, %237
  %249 = phi i32 [ %247, %242 ], [ 0, %237 ]
  %250 = zext i32 %238 to i64
  %251 = getelementptr inbounds i32, i32* %231, i64 %250
  store i32 %249, i32* %251, align 4, !tbaa !518
  %252 = add nuw nsw i32 %238, 1
  %253 = icmp ult i32 %238, 31
  br i1 %253, label %237, label %__builtin_spriv_OpJointMatrixLoadINTEL_PackedB_PackedB_16x32_i16_32_global_v8i8_pi32_i32.25.exit

__builtin_spriv_OpJointMatrixLoadINTEL_PackedB_PackedB_16x32_i16_32_global_v8i8_pi32_i32.25.exit: ; preds = %248
  %254 = add nuw nsw i32 %169, 1, !spirv.Decorations !522
  br label %168

255:                                              ; preds = %168
  %.fca.0.extract = extractvalue [2 x <64 x float>] %157, 0
  %.fca.0.gep = getelementptr inbounds [2 x <64 x float>], [2 x <64 x float>]* %43, i32 0, i32 0
  store <64 x float> %.fca.0.extract, <64 x float>* %.fca.0.gep, align 256
  %.fca.1.extract = extractvalue [2 x <64 x float>] %157, 1
  %.fca.1.gep = getelementptr inbounds [2 x <64 x float>], [2 x <64 x float>]* %43, i32 0, i32 1
  store <64 x float> %.fca.1.extract, <64 x float>* %.fca.1.gep, align 256
  %256 = bitcast [2 x <64 x float>]* %43 to i8*
  %.ascast.i = bitcast float addrspace(1)* %105 to i8 addrspace(1)*
  %localSize58 = extractelement <3 x i32> %localSize, i32 0
  %localSize59 = extractelement <3 x i32> %localSize, i32 1
  %257 = mul i32 %localSize59, %localSize58
  %localSize60 = extractelement <3 x i32> %localSize, i32 2
  %258 = mul i32 %257, %localSize60
  %259 = add i32 8, -1
  %260 = and i32 %259, %258
  %261 = icmp eq i32 %260, 0
  br i1 %261, label %275, label %262

262:                                              ; preds = %255
  %localIdZ61 = zext i16 %localIdZ to i32
  %localSize62 = extractelement <3 x i32> %localSize, i32 1
  %263 = mul i32 %localSize62, %localIdZ61
  %localIdY63 = zext i16 %localIdY to i32
  %264 = add i32 %263, %localIdY63
  %localSize64 = extractelement <3 x i32> %localSize, i32 0
  %265 = mul i32 %264, %localSize64
  %localIdX65 = zext i16 %localIdX to i32
  %266 = add i32 %265, %localIdX65
  %267 = udiv i32 %266, 8
  %localSize66 = extractelement <3 x i32> %localSize, i32 0
  %localSize67 = extractelement <3 x i32> %localSize, i32 1
  %268 = mul i32 %localSize67, %localSize66
  %localSize68 = extractelement <3 x i32> %localSize, i32 2
  %269 = mul i32 %268, %localSize68
  %270 = add i32 %269, -1
  %271 = add i32 %270, 8
  %272 = udiv i32 %271, 8
  %273 = add i32 %272, -1
  %274 = icmp ult i32 %267, %273
  br i1 %274, label %275, label %_Z18get_sub_group_sizev.exit.i

275:                                              ; preds = %262, %255
  br label %_Z18get_sub_group_sizev.exit.i

_Z18get_sub_group_sizev.exit.i:                   ; preds = %275, %262
  %276 = phi i32 [ 8, %275 ], [ %260, %262 ]
  %277 = bitcast i8 addrspace(1)* %.ascast.i to i32 addrspace(1)*
  %simdLaneId1671 = call i16 @llvm.genx.GenISA.simdLaneId()
  %simdLaneId72 = zext i16 %simdLaneId1671 to i32
  %278 = sdiv i32 %276, 32
  %279 = bitcast i8* %256 to i32*
  %280 = freeze i32 %simdLaneId72
  %281 = sdiv i32 %280, 32
  %282 = mul i32 %281, 32
  %283 = sub i32 %280, %282
  %284 = sext i32 %283 to i64
  br label %285

285:                                              ; preds = %298, %_Z18get_sub_group_sizev.exit.i
  %286 = phi i32 [ 0, %_Z18get_sub_group_sizev.exit.i ], [ %299, %298 ]
  %287 = mul nsw i32 %286, %278
  %288 = add nsw i32 %287, %281
  %289 = icmp slt i32 %288, 32
  br i1 %289, label %290, label %298

290:                                              ; preds = %285
  %291 = zext i32 %286 to i64
  %292 = getelementptr inbounds i32, i32* %279, i64 %291
  %293 = load i32, i32* %292, align 4, !tbaa !518
  %294 = sext i32 %288 to i64
  %295 = mul nsw i64 %294, 64
  %296 = add nsw i64 %295, %284
  %297 = getelementptr inbounds i32, i32 addrspace(1)* %277, i64 %296
  store i32 %293, i32 addrspace(1)* %297, align 4, !tbaa !518
  br label %298

298:                                              ; preds = %290, %285
  %299 = add nuw nsw i32 %286, 1
  %300 = icmp ult i32 %286, 127
  br i1 %300, label %285, label %__builtin_spriv_OpJointMatrixStoreINTEL_Accumulator_RowMajor_32x32_i32_128_global_pi64_v8i8.26.exit

__builtin_spriv_OpJointMatrixStoreINTEL_Accumulator_RowMajor_32x32_i32_128_global_pi64_v8i8.26.exit: ; preds = %298
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
attributes #6 = { nounwind }

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
