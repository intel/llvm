; ------------------------------------------------
; OCL_asm40d739ce99ad8281_0068_Unify_after_WIFuncResolution.ll
; LLVM major version: 14
; ------------------------------------------------
target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v16:16:16-v24:32:32-v32:32:32-v48:64:64-v64:64:64-v96:128:128-v128:128:128-v192:256:256-v256:256:256-v512:512:512-v1024:1024:1024-n8:16:32"
target triple = "spir64-unknown-unknown"

%"class.sycl::_V1::ext::oneapi::bfloat16" = type { i16 }
%"class.sycl::_V1::range" = type { %"class.sycl::_V1::detail::array" }
%"class.sycl::_V1::detail::array" = type { [2 x i64] }

; Function Attrs: convergent nounwind
define spir_kernel void @_ZTS7imatrixIfLm32ELm32ELm16EE(float addrspace(1)* align 4 %0, %"class.sycl::_V1::range"* byval(%"class.sycl::_V1::range") align 8 %1, %"class.sycl::_V1::range"* byval(%"class.sycl::_V1::range") align 8 %2, i64 %3, %"class.sycl::_V1::ext::oneapi::bfloat16" addrspace(1)* align 2 %4, %"class.sycl::_V1::range"* byval(%"class.sycl::_V1::range") align 8 %5, %"class.sycl::_V1::range"* byval(%"class.sycl::_V1::range") align 8 %6, %"class.sycl::_V1::ext::oneapi::bfloat16" addrspace(1)* align 2 %7, %"class.sycl::_V1::range"* byval(%"class.sycl::_V1::range") align 8 %8, %"class.sycl::_V1::range"* byval(%"class.sycl::_V1::range") align 8 %9, <8 x i32> %r0, <8 x i32> %payloadHeader, <3 x i32> %localSize, <3 x i32> %enqueuedLocalSize, i16 %localIdX, i16 %localIdY, i16 %localIdZ, i8* %privateBase, i64 %const_reg_qword, i64 %const_reg_qword1, i64 %const_reg_qword2, i64 %const_reg_qword3, i64 %const_reg_qword4, i64 %const_reg_qword5, i64 %const_reg_qword6, i64 %const_reg_qword7, i64 %const_reg_qword8, i64 %const_reg_qword9, i64 %const_reg_qword10, i64 %const_reg_qword11, i32 %bufferOffset, i32 %bufferOffset12, i32 %bufferOffset13) #0 {
  %11 = alloca <32 x i32>, align 128
  %12 = alloca <32 x i32>, align 128
  %13 = alloca [2 x <64 x float>], align 256
  %14 = alloca [2 x <64 x float>], align 256
  %15 = bitcast %"class.sycl::_V1::range"* %1 to i64*
  %16 = getelementptr inbounds i64, i64* %15, i64 1
  %17 = load i64, i64* %16, align 8
  %18 = bitcast %"class.sycl::_V1::range"* %2 to i64*
  %19 = load i64, i64* %18, align 8
  %20 = bitcast %"class.sycl::_V1::range"* %2 to i64*
  %21 = getelementptr inbounds i64, i64* %20, i64 1
  %22 = load i64, i64* %21, align 8
  %23 = mul i64 %19, %17
  %24 = getelementptr float, float addrspace(1)* %0, i64 %23
  %25 = getelementptr float, float addrspace(1)* %24, i64 %22
  %26 = bitcast %"class.sycl::_V1::range"* %5 to i64*
  %27 = getelementptr inbounds i64, i64* %26, i64 1
  %28 = load i64, i64* %27, align 8
  %29 = bitcast %"class.sycl::_V1::range"* %6 to i64*
  %30 = load i64, i64* %29, align 8
  %31 = bitcast %"class.sycl::_V1::range"* %6 to i64*
  %32 = getelementptr inbounds i64, i64* %31, i64 1
  %33 = load i64, i64* %32, align 8
  %34 = mul i64 %30, %28
  %35 = getelementptr %"class.sycl::_V1::ext::oneapi::bfloat16", %"class.sycl::_V1::ext::oneapi::bfloat16" addrspace(1)* %4, i64 %34
  %36 = getelementptr %"class.sycl::_V1::ext::oneapi::bfloat16", %"class.sycl::_V1::ext::oneapi::bfloat16" addrspace(1)* %35, i64 %33
  %37 = bitcast %"class.sycl::_V1::range"* %8 to i64*
  %38 = getelementptr inbounds i64, i64* %37, i64 1
  %39 = load i64, i64* %38, align 8
  %40 = bitcast %"class.sycl::_V1::range"* %9 to i64*
  %41 = load i64, i64* %40, align 8
  %42 = bitcast %"class.sycl::_V1::range"* %9 to i64*
  %43 = getelementptr inbounds i64, i64* %42, i64 1
  %44 = load i64, i64* %43, align 8
  %45 = mul i64 %41, %39
  %46 = getelementptr %"class.sycl::_V1::ext::oneapi::bfloat16", %"class.sycl::_V1::ext::oneapi::bfloat16" addrspace(1)* %7, i64 %45
  %47 = getelementptr %"class.sycl::_V1::ext::oneapi::bfloat16", %"class.sycl::_V1::ext::oneapi::bfloat16" addrspace(1)* %46, i64 %44
  %cmpDim = icmp eq i32 1, 0
  %tmpOffsetR0 = select i1 %cmpDim, i32 1, i32 5
  %offsetR0 = add i32 1, %tmpOffsetR0
  %groupId = extractelement <8 x i32> %r0, i32 %offsetR0
  %enqueuedLocalSize14 = extractelement <3 x i32> %enqueuedLocalSize, i32 1
  %48 = mul i32 %enqueuedLocalSize14, %groupId
  %localIdY15 = zext i16 %localIdY to i32
  %49 = icmp ult i32 %localIdY15, 65536
  call void @llvm.assume(i1 %49) #4
  %50 = add i32 %localIdY15, %48
  %globalOffset = extractelement <8 x i32> %payloadHeader, i32 1
  %51 = add i32 %50, %globalOffset
  %52 = zext i32 %51 to i64
  %53 = icmp ult i64 %52, 2147483648
  call void @llvm.assume(i1 %53)
  %cmpDim16 = icmp eq i32 0, 0
  %tmpOffsetR017 = select i1 %cmpDim16, i32 1, i32 5
  %offsetR018 = add i32 0, %tmpOffsetR017
  %groupId19 = extractelement <8 x i32> %r0, i32 %offsetR018
  %enqueuedLocalSize20 = extractelement <3 x i32> %enqueuedLocalSize, i32 0
  %54 = mul i32 %enqueuedLocalSize20, %groupId19
  %localIdX21 = zext i16 %localIdX to i32
  %55 = icmp ult i32 %localIdX21, 65536
  call void @llvm.assume(i1 %55) #4
  %56 = add i32 %localIdX21, %54
  %globalOffset22 = extractelement <8 x i32> %payloadHeader, i32 0
  %57 = add i32 %56, %globalOffset22
  %58 = zext i32 %57 to i64
  %59 = icmp ult i64 %58, 2147483648
  call void @llvm.assume(i1 %59)
  %localIdY23 = zext i16 %localIdY to i32
  %60 = icmp ult i32 %localIdY23, 65536
  call void @llvm.assume(i1 %60) #4
  %61 = zext i32 %localIdY23 to i64
  %62 = icmp ult i64 %61, 2147483648
  call void @llvm.assume(i1 %62)
  %63 = sub nsw i64 %52, %61, !spirv.Decorations !475
  %localIdX24 = zext i16 %localIdX to i32
  %64 = icmp ult i32 %localIdX24, 65536
  call void @llvm.assume(i1 %64) #4
  %65 = zext i32 %localIdX24 to i64
  %66 = icmp ult i64 %65, 2147483648
  call void @llvm.assume(i1 %66)
  %67 = sub nsw i64 %58, %65, !spirv.Decorations !475
  %68 = add i64 %23, %22
  %69 = sub i64 0, %68
  %70 = getelementptr inbounds float, float addrspace(1)* %25, i64 %69
  %71 = shl nsw i64 %63, 11, !spirv.Decorations !475
  %72 = getelementptr inbounds float, float addrspace(1)* %70, i64 %71
  %73 = udiv i64 %67, %3
  %74 = shl i64 %73, 5
  %75 = getelementptr inbounds float, float addrspace(1)* %72, i64 %74
  %76 = bitcast [2 x <64 x float>]* %14 to i8*
  %.ascast.i3 = bitcast float addrspace(1)* %75 to i8 addrspace(1)*
  %localSize25 = extractelement <3 x i32> %localSize, i32 0
  %localSize26 = extractelement <3 x i32> %localSize, i32 1
  %77 = mul i32 %localSize26, %localSize25
  %localSize27 = extractelement <3 x i32> %localSize, i32 2
  %78 = mul i32 %77, %localSize27
  %79 = call spir_func i32 @__builtin_IB_get_simd_size() #5
  %80 = add i32 %79, -1
  %81 = and i32 %80, %78
  %82 = icmp eq i32 %81, 0
  br i1 %82, label %99, label %83

83:                                               ; preds = %10
  %localIdZ28 = zext i16 %localIdZ to i32
  %localSize29 = extractelement <3 x i32> %localSize, i32 1
  %84 = mul i32 %localSize29, %localIdZ28
  %localIdY30 = zext i16 %localIdY to i32
  %85 = add i32 %84, %localIdY30
  %localSize31 = extractelement <3 x i32> %localSize, i32 0
  %86 = mul i32 %85, %localSize31
  %localIdX32 = zext i16 %localIdX to i32
  %87 = add i32 %86, %localIdX32
  %88 = call spir_func i32 @__builtin_IB_get_simd_size() #5
  %89 = udiv i32 %87, %88
  %localSize33 = extractelement <3 x i32> %localSize, i32 0
  %localSize34 = extractelement <3 x i32> %localSize, i32 1
  %90 = mul i32 %localSize34, %localSize33
  %localSize35 = extractelement <3 x i32> %localSize, i32 2
  %91 = mul i32 %90, %localSize35
  %92 = call spir_func i32 @__builtin_IB_get_simd_size() #5
  %93 = add i32 %91, -1
  %94 = add i32 %93, %92
  %95 = call spir_func i32 @__builtin_IB_get_simd_size() #5
  %96 = udiv i32 %94, %95
  %97 = add i32 %96, -1
  %98 = icmp ult i32 %89, %97
  br i1 %98, label %99, label %_Z18get_sub_group_sizev.exit.i4

99:                                               ; preds = %83, %10
  %100 = call spir_func i32 @__builtin_IB_get_simd_size() #5
  br label %_Z18get_sub_group_sizev.exit.i4

_Z18get_sub_group_sizev.exit.i4:                  ; preds = %99, %83
  %101 = phi i32 [ %100, %99 ], [ %81, %83 ]
  %102 = bitcast i8 addrspace(1)* %.ascast.i3 to i32 addrspace(1)*
  %103 = call spir_func i32 @__builtin_IB_get_simd_id() #5
  %104 = sdiv i32 %101, 32
  %105 = bitcast i8* %76 to i32*
  %106 = freeze i32 %103
  %107 = sdiv i32 %106, 32
  %108 = mul i32 %107, 32
  %109 = sub i32 %106, %108
  %110 = sext i32 %109 to i64
  br label %111

111:                                              ; preds = %122, %_Z18get_sub_group_sizev.exit.i4
  %112 = phi i32 [ 0, %_Z18get_sub_group_sizev.exit.i4 ], [ %126, %122 ]
  %113 = mul nsw i32 %112, %104
  %114 = add nsw i32 %113, %107
  %115 = icmp slt i32 %114, 32
  br i1 %115, label %116, label %122

116:                                              ; preds = %111
  %117 = sext i32 %114 to i64
  %118 = mul nsw i64 %117, 64
  %119 = add nsw i64 %118, %110
  %120 = getelementptr inbounds i32, i32 addrspace(1)* %102, i64 %119
  %121 = load i32, i32 addrspace(1)* %120, align 4, !tbaa !477
  br label %122

122:                                              ; preds = %116, %111
  %123 = phi i32 [ %121, %116 ], [ 0, %111 ]
  %124 = zext i32 %112 to i64
  %125 = getelementptr inbounds i32, i32* %105, i64 %124
  store i32 %123, i32* %125, align 4, !tbaa !477
  %126 = add nuw nsw i32 %112, 1
  %127 = icmp ult i32 %112, 127
  br i1 %127, label %111, label %__builtin_spriv_OpJointMatrixLoadINTEL_Accumulator_RowMajor_32x32_i32_128_global_v8i8_pi32_i32.23.exit

__builtin_spriv_OpJointMatrixLoadINTEL_Accumulator_RowMajor_32x32_i32_128_global_v8i8_pi32_i32.23.exit: ; preds = %122
  %128 = bitcast [2 x <64 x float>]* %14 to <64 x float>*
  %129 = load <64 x float>, <64 x float>* %128, align 256
  %130 = getelementptr <64 x float>, <64 x float>* %128, i32 1
  %131 = load <64 x float>, <64 x float>* %130, align 256
  %132 = insertvalue [2 x <64 x float>] undef, <64 x float> %129, 0
  %133 = insertvalue [2 x <64 x float>] %132, <64 x float> %131, 1
  %134 = add i64 %34, %33
  %135 = sub i64 0, %134
  %136 = getelementptr inbounds %"class.sycl::_V1::ext::oneapi::bfloat16", %"class.sycl::_V1::ext::oneapi::bfloat16" addrspace(1)* %36, i64 %135
  %137 = shl nsw i64 %63, 10, !spirv.Decorations !475
  %138 = getelementptr inbounds %"class.sycl::_V1::ext::oneapi::bfloat16", %"class.sycl::_V1::ext::oneapi::bfloat16" addrspace(1)* %136, i64 %137
  %139 = add i64 %45, %44
  %140 = sub i64 0, %139
  %141 = getelementptr inbounds %"class.sycl::_V1::ext::oneapi::bfloat16", %"class.sycl::_V1::ext::oneapi::bfloat16" addrspace(1)* %47, i64 %140
  %142 = shl i64 %73, 6
  %143 = getelementptr %"class.sycl::_V1::ext::oneapi::bfloat16", %"class.sycl::_V1::ext::oneapi::bfloat16" addrspace(1)* %141, i64 %142
  br label %144

144:                                              ; preds = %__builtin_spriv_OpJointMatrixLoadINTEL_PackedB_PackedB_16x32_i16_32_global_v8i8_pi32_i32.25.exit, %__builtin_spriv_OpJointMatrixLoadINTEL_Accumulator_RowMajor_32x32_i32_128_global_v8i8_pi32_i32.23.exit
  %145 = phi i32 [ 0, %__builtin_spriv_OpJointMatrixLoadINTEL_Accumulator_RowMajor_32x32_i32_128_global_v8i8_pi32_i32.23.exit ], [ %241, %__builtin_spriv_OpJointMatrixLoadINTEL_PackedB_PackedB_16x32_i16_32_global_v8i8_pi32_i32.25.exit ]
  %146 = icmp ult i32 %145, 2
  br i1 %146, label %147, label %242

147:                                              ; preds = %144
  %148 = shl nuw nsw i32 %145, 4, !spirv.Decorations !481
  %149 = zext i32 %148 to i64
  %150 = getelementptr inbounds %"class.sycl::_V1::ext::oneapi::bfloat16", %"class.sycl::_V1::ext::oneapi::bfloat16" addrspace(1)* %138, i64 %149
  %151 = bitcast <32 x i32>* %12 to i8*
  %.ascast.i5 = bitcast %"class.sycl::_V1::ext::oneapi::bfloat16" addrspace(1)* %150 to i8 addrspace(1)*
  %localSize36 = extractelement <3 x i32> %localSize, i32 0
  %localSize37 = extractelement <3 x i32> %localSize, i32 1
  %152 = mul i32 %localSize37, %localSize36
  %localSize38 = extractelement <3 x i32> %localSize, i32 2
  %153 = mul i32 %152, %localSize38
  %154 = call spir_func i32 @__builtin_IB_get_simd_size() #5
  %155 = add i32 %154, -1
  %156 = and i32 %155, %153
  %157 = icmp eq i32 %156, 0
  br i1 %157, label %174, label %158

158:                                              ; preds = %147
  %localIdZ39 = zext i16 %localIdZ to i32
  %localSize40 = extractelement <3 x i32> %localSize, i32 1
  %159 = mul i32 %localSize40, %localIdZ39
  %localIdY41 = zext i16 %localIdY to i32
  %160 = add i32 %159, %localIdY41
  %localSize42 = extractelement <3 x i32> %localSize, i32 0
  %161 = mul i32 %160, %localSize42
  %localIdX43 = zext i16 %localIdX to i32
  %162 = add i32 %161, %localIdX43
  %163 = call spir_func i32 @__builtin_IB_get_simd_size() #5
  %164 = udiv i32 %162, %163
  %localSize44 = extractelement <3 x i32> %localSize, i32 0
  %localSize45 = extractelement <3 x i32> %localSize, i32 1
  %165 = mul i32 %localSize45, %localSize44
  %localSize46 = extractelement <3 x i32> %localSize, i32 2
  %166 = mul i32 %165, %localSize46
  %167 = call spir_func i32 @__builtin_IB_get_simd_size() #5
  %168 = add i32 %166, -1
  %169 = add i32 %168, %167
  %170 = call spir_func i32 @__builtin_IB_get_simd_size() #5
  %171 = udiv i32 %169, %170
  %172 = add i32 %171, -1
  %173 = icmp ult i32 %164, %172
  br i1 %173, label %174, label %_Z18get_sub_group_sizev.exit.i6

174:                                              ; preds = %158, %147
  %175 = call spir_func i32 @__builtin_IB_get_simd_size() #5
  br label %_Z18get_sub_group_sizev.exit.i6

_Z18get_sub_group_sizev.exit.i6:                  ; preds = %174, %158
  %176 = bitcast i8* %151 to i32*
  %177 = bitcast i8 addrspace(1)* %.ascast.i5 to i32 addrspace(1)*
  br label %178

178:                                              ; preds = %178, %_Z18get_sub_group_sizev.exit.i6
  %179 = phi i32 [ 0, %_Z18get_sub_group_sizev.exit.i6 ], [ %185, %178 ]
  %180 = zext i32 %179 to i64
  %181 = mul nsw i64 16, %180
  %182 = getelementptr inbounds i32, i32 addrspace(1)* %177, i64 %181
  %183 = call spir_func i32 @__builtin_IB_simd_block_read_1_global(i32 addrspace(1)* noundef %182) #5
  %184 = getelementptr inbounds i32, i32* %176, i64 %180
  store i32 %183, i32* %184, align 4, !tbaa !477
  %185 = add nuw nsw i32 %179, 1
  %186 = icmp ult i32 %179, 31
  br i1 %186, label %178, label %__builtin_spriv_OpJointMatrixLoadINTEL_PackedA_RowMajor_32x16_i16_32_global_v8i8_pi32_i32.24.exit

__builtin_spriv_OpJointMatrixLoadINTEL_PackedA_RowMajor_32x16_i16_32_global_v8i8_pi32_i32.24.exit: ; preds = %178
  %187 = shl nuw nsw i64 %149, 6, !spirv.Decorations !481
  %188 = getelementptr %"class.sycl::_V1::ext::oneapi::bfloat16", %"class.sycl::_V1::ext::oneapi::bfloat16" addrspace(1)* %143, i64 %187
  %189 = bitcast <32 x i32>* %11 to i8*
  %.ascast.i7 = bitcast %"class.sycl::_V1::ext::oneapi::bfloat16" addrspace(1)* %188 to i8 addrspace(1)*
  %localSize47 = extractelement <3 x i32> %localSize, i32 0
  %localSize48 = extractelement <3 x i32> %localSize, i32 1
  %190 = mul i32 %localSize48, %localSize47
  %localSize49 = extractelement <3 x i32> %localSize, i32 2
  %191 = mul i32 %190, %localSize49
  %192 = call spir_func i32 @__builtin_IB_get_simd_size() #5
  %193 = add i32 %192, -1
  %194 = and i32 %193, %191
  %195 = icmp eq i32 %194, 0
  br i1 %195, label %212, label %196

196:                                              ; preds = %__builtin_spriv_OpJointMatrixLoadINTEL_PackedA_RowMajor_32x16_i16_32_global_v8i8_pi32_i32.24.exit
  %localIdZ50 = zext i16 %localIdZ to i32
  %localSize51 = extractelement <3 x i32> %localSize, i32 1
  %197 = mul i32 %localSize51, %localIdZ50
  %localIdY52 = zext i16 %localIdY to i32
  %198 = add i32 %197, %localIdY52
  %localSize53 = extractelement <3 x i32> %localSize, i32 0
  %199 = mul i32 %198, %localSize53
  %localIdX54 = zext i16 %localIdX to i32
  %200 = add i32 %199, %localIdX54
  %201 = call spir_func i32 @__builtin_IB_get_simd_size() #5
  %202 = udiv i32 %200, %201
  %localSize55 = extractelement <3 x i32> %localSize, i32 0
  %localSize56 = extractelement <3 x i32> %localSize, i32 1
  %203 = mul i32 %localSize56, %localSize55
  %localSize57 = extractelement <3 x i32> %localSize, i32 2
  %204 = mul i32 %203, %localSize57
  %205 = call spir_func i32 @__builtin_IB_get_simd_size() #5
  %206 = add i32 %204, -1
  %207 = add i32 %206, %205
  %208 = call spir_func i32 @__builtin_IB_get_simd_size() #5
  %209 = udiv i32 %207, %208
  %210 = add i32 %209, -1
  %211 = icmp ult i32 %202, %210
  br i1 %211, label %212, label %_Z18get_sub_group_sizev.exit.i8

212:                                              ; preds = %196, %__builtin_spriv_OpJointMatrixLoadINTEL_PackedA_RowMajor_32x16_i16_32_global_v8i8_pi32_i32.24.exit
  %213 = call spir_func i32 @__builtin_IB_get_simd_size() #5
  br label %_Z18get_sub_group_sizev.exit.i8

_Z18get_sub_group_sizev.exit.i8:                  ; preds = %212, %196
  %214 = phi i32 [ %213, %212 ], [ %194, %196 ]
  %215 = bitcast i8 addrspace(1)* %.ascast.i7 to i32 addrspace(1)*
  %216 = call spir_func i32 @__builtin_IB_get_simd_id() #5
  %217 = sdiv i32 %214, 32
  %218 = bitcast i8* %189 to i32*
  %219 = freeze i32 %216
  %220 = sdiv i32 %219, 32
  %221 = mul i32 %220, 32
  %222 = sub i32 %219, %221
  %223 = sext i32 %222 to i64
  br label %224

224:                                              ; preds = %235, %_Z18get_sub_group_sizev.exit.i8
  %225 = phi i32 [ 0, %_Z18get_sub_group_sizev.exit.i8 ], [ %239, %235 ]
  %226 = mul nsw i32 %225, %217
  %227 = add nsw i32 %226, %220
  %228 = icmp slt i32 %227, 8
  br i1 %228, label %229, label %235

229:                                              ; preds = %224
  %230 = sext i32 %227 to i64
  %231 = mul nsw i64 64, %230
  %232 = add nsw i64 %231, %223
  %233 = getelementptr inbounds i32, i32 addrspace(1)* %215, i64 %232
  %234 = load i32, i32 addrspace(1)* %233, align 4, !tbaa !477
  br label %235

235:                                              ; preds = %229, %224
  %236 = phi i32 [ %234, %229 ], [ 0, %224 ]
  %237 = zext i32 %225 to i64
  %238 = getelementptr inbounds i32, i32* %218, i64 %237
  store i32 %236, i32* %238, align 4, !tbaa !477
  %239 = add nuw nsw i32 %225, 1
  %240 = icmp ult i32 %225, 31
  br i1 %240, label %224, label %__builtin_spriv_OpJointMatrixLoadINTEL_PackedB_PackedB_16x32_i16_32_global_v8i8_pi32_i32.25.exit

__builtin_spriv_OpJointMatrixLoadINTEL_PackedB_PackedB_16x32_i16_32_global_v8i8_pi32_i32.25.exit: ; preds = %235
  %241 = add nuw nsw i32 %145, 1, !spirv.Decorations !481
  br label %144

242:                                              ; preds = %144
  %.fca.0.extract = extractvalue [2 x <64 x float>] %133, 0
  %.fca.0.gep = getelementptr inbounds [2 x <64 x float>], [2 x <64 x float>]* %13, i32 0, i32 0
  store <64 x float> %.fca.0.extract, <64 x float>* %.fca.0.gep, align 256
  %.fca.1.extract = extractvalue [2 x <64 x float>] %133, 1
  %.fca.1.gep = getelementptr inbounds [2 x <64 x float>], [2 x <64 x float>]* %13, i32 0, i32 1
  store <64 x float> %.fca.1.extract, <64 x float>* %.fca.1.gep, align 256
  %243 = bitcast [2 x <64 x float>]* %13 to i8*
  %.ascast.i = bitcast float addrspace(1)* %75 to i8 addrspace(1)*
  %localSize58 = extractelement <3 x i32> %localSize, i32 0
  %localSize59 = extractelement <3 x i32> %localSize, i32 1
  %244 = mul i32 %localSize59, %localSize58
  %localSize60 = extractelement <3 x i32> %localSize, i32 2
  %245 = mul i32 %244, %localSize60
  %246 = call spir_func i32 @__builtin_IB_get_simd_size() #5
  %247 = add i32 %246, -1
  %248 = and i32 %247, %245
  %249 = icmp eq i32 %248, 0
  br i1 %249, label %266, label %250

250:                                              ; preds = %242
  %localIdZ61 = zext i16 %localIdZ to i32
  %localSize62 = extractelement <3 x i32> %localSize, i32 1
  %251 = mul i32 %localSize62, %localIdZ61
  %localIdY63 = zext i16 %localIdY to i32
  %252 = add i32 %251, %localIdY63
  %localSize64 = extractelement <3 x i32> %localSize, i32 0
  %253 = mul i32 %252, %localSize64
  %localIdX65 = zext i16 %localIdX to i32
  %254 = add i32 %253, %localIdX65
  %255 = call spir_func i32 @__builtin_IB_get_simd_size() #5
  %256 = udiv i32 %254, %255
  %localSize66 = extractelement <3 x i32> %localSize, i32 0
  %localSize67 = extractelement <3 x i32> %localSize, i32 1
  %257 = mul i32 %localSize67, %localSize66
  %localSize68 = extractelement <3 x i32> %localSize, i32 2
  %258 = mul i32 %257, %localSize68
  %259 = call spir_func i32 @__builtin_IB_get_simd_size() #5
  %260 = add i32 %258, -1
  %261 = add i32 %260, %259
  %262 = call spir_func i32 @__builtin_IB_get_simd_size() #5
  %263 = udiv i32 %261, %262
  %264 = add i32 %263, -1
  %265 = icmp ult i32 %256, %264
  br i1 %265, label %266, label %_Z18get_sub_group_sizev.exit.i

266:                                              ; preds = %250, %242
  %267 = call spir_func i32 @__builtin_IB_get_simd_size() #5
  br label %_Z18get_sub_group_sizev.exit.i

_Z18get_sub_group_sizev.exit.i:                   ; preds = %266, %250
  %268 = phi i32 [ %267, %266 ], [ %248, %250 ]
  %269 = bitcast i8 addrspace(1)* %.ascast.i to i32 addrspace(1)*
  %270 = call spir_func i32 @__builtin_IB_get_simd_id() #5
  %271 = sdiv i32 %268, 32
  %272 = bitcast i8* %243 to i32*
  %273 = freeze i32 %270
  %274 = sdiv i32 %273, 32
  %275 = mul i32 %274, 32
  %276 = sub i32 %273, %275
  %277 = sext i32 %276 to i64
  br label %278

278:                                              ; preds = %291, %_Z18get_sub_group_sizev.exit.i
  %279 = phi i32 [ 0, %_Z18get_sub_group_sizev.exit.i ], [ %292, %291 ]
  %280 = mul nsw i32 %279, %271
  %281 = add nsw i32 %280, %274
  %282 = icmp slt i32 %281, 32
  br i1 %282, label %283, label %291

283:                                              ; preds = %278
  %284 = zext i32 %279 to i64
  %285 = getelementptr inbounds i32, i32* %272, i64 %284
  %286 = load i32, i32* %285, align 4, !tbaa !477
  %287 = sext i32 %281 to i64
  %288 = mul nsw i64 %287, 64
  %289 = add nsw i64 %288, %277
  %290 = getelementptr inbounds i32, i32 addrspace(1)* %269, i64 %289
  store i32 %286, i32 addrspace(1)* %290, align 4, !tbaa !477
  br label %291

291:                                              ; preds = %283, %278
  %292 = add nuw nsw i32 %279, 1
  %293 = icmp ult i32 %279, 127
  br i1 %293, label %278, label %__builtin_spriv_OpJointMatrixStoreINTEL_Accumulator_RowMajor_32x32_i32_128_global_pi64_v8i8.26.exit

__builtin_spriv_OpJointMatrixStoreINTEL_Accumulator_RowMajor_32x32_i32_128_global_pi64_v8i8.26.exit: ; preds = %291
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

attributes #0 = { convergent nounwind "less-precise-fpmad"="true" }
attributes #1 = { inaccessiblememonly nofree nosync nounwind willreturn }
attributes #2 = { convergent "frame-pointer"="none" "no-trapping-math"="true" "stack-protector-buffer-size"="8" }
attributes #3 = { convergent mustprogress nofree nounwind readnone willreturn "frame-pointer"="none" "no-trapping-math"="true" "stack-protector-buffer-size"="8" }
attributes #4 = { nounwind }
attributes #5 = { convergent nounwind }

!spirv.MemoryModel = !{!0}
!spirv.Source = !{!1}
!spirv.Generator = !{!2}
!igc.functions = !{!3}
!IGCMetadata = !{!42}
!opencl.ocl.version = !{!472, !472, !472, !472, !472}
!opencl.spir.version = !{!472, !472, !472, !472, !472}
!llvm.ident = !{!473, !473, !473, !473, !473}
!llvm.module.flags = !{!474}

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
!42 = !{!"ModuleMD", !43, !44, !141, !267, !298, !299, !303, !306, !307, !308, !343, !369, !382, !383, !384, !399, !400, !401, !402, !403, !404, !405, !406, !407, !408, !412, !413, !420, !421, !422, !423, !424, !425, !426, !427, !428, !429, !430, !431, !433, !437, !438, !439, !440, !441, !442, !443, !444, !445, !446, !447, !448, !449, !450, !451, !452, !453, !191, !454, !455, !456, !458, !461, !462, !463, !465, !466, !467}
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
!143 = !{!"FuncMDValue[0]", !144, !145, !149, !150, !151, !152, !153, !154, !155, !177, !183, !184, !185, !186, !187, !188, !189, !190, !191, !192, !193, !194, !195, !196, !197, !198, !199, !210, !221, !232, !243, !254, !265, !266}
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
!177 = !{!"resAllocMD", !178, !179, !180, !181, !182}
!178 = !{!"uavsNumType", i32 0}
!179 = !{!"srvsNumType", i32 0}
!180 = !{!"samplersNumType", i32 0}
!181 = !{!"argAllocMDList"}
!182 = !{!"inlineSamplersMD"}
!183 = !{!"maxByteOffsets"}
!184 = !{!"IsInitializer", i1 false}
!185 = !{!"IsFinalizer", i1 false}
!186 = !{!"CompiledSubGroupsNumber", i32 0}
!187 = !{!"hasInlineVmeSamplers", i1 false}
!188 = !{!"localSize", i32 0}
!189 = !{!"localIDPresent", i1 false}
!190 = !{!"groupIDPresent", i1 false}
!191 = !{!"privateMemoryPerWI", i32 0}
!192 = !{!"prevFPOffset", i32 0}
!193 = !{!"globalIDPresent", i1 false}
!194 = !{!"hasSyncRTCalls", i1 false}
!195 = !{!"hasNonKernelArgLoad", i1 false}
!196 = !{!"hasNonKernelArgStore", i1 false}
!197 = !{!"hasNonKernelArgAtomic", i1 false}
!198 = !{!"UserAnnotations"}
!199 = !{!"m_OpenCLArgAddressSpaces", !200, !201, !202, !203, !204, !205, !206, !207, !208, !209}
!200 = !{!"m_OpenCLArgAddressSpacesVec[0]", i32 1}
!201 = !{!"m_OpenCLArgAddressSpacesVec[1]", i32 0}
!202 = !{!"m_OpenCLArgAddressSpacesVec[2]", i32 0}
!203 = !{!"m_OpenCLArgAddressSpacesVec[3]", i32 0}
!204 = !{!"m_OpenCLArgAddressSpacesVec[4]", i32 1}
!205 = !{!"m_OpenCLArgAddressSpacesVec[5]", i32 0}
!206 = !{!"m_OpenCLArgAddressSpacesVec[6]", i32 0}
!207 = !{!"m_OpenCLArgAddressSpacesVec[7]", i32 1}
!208 = !{!"m_OpenCLArgAddressSpacesVec[8]", i32 0}
!209 = !{!"m_OpenCLArgAddressSpacesVec[9]", i32 0}
!210 = !{!"m_OpenCLArgAccessQualifiers", !211, !212, !213, !214, !215, !216, !217, !218, !219, !220}
!211 = !{!"m_OpenCLArgAccessQualifiersVec[0]", !"none"}
!212 = !{!"m_OpenCLArgAccessQualifiersVec[1]", !"none"}
!213 = !{!"m_OpenCLArgAccessQualifiersVec[2]", !"none"}
!214 = !{!"m_OpenCLArgAccessQualifiersVec[3]", !"none"}
!215 = !{!"m_OpenCLArgAccessQualifiersVec[4]", !"none"}
!216 = !{!"m_OpenCLArgAccessQualifiersVec[5]", !"none"}
!217 = !{!"m_OpenCLArgAccessQualifiersVec[6]", !"none"}
!218 = !{!"m_OpenCLArgAccessQualifiersVec[7]", !"none"}
!219 = !{!"m_OpenCLArgAccessQualifiersVec[8]", !"none"}
!220 = !{!"m_OpenCLArgAccessQualifiersVec[9]", !"none"}
!221 = !{!"m_OpenCLArgTypes", !222, !223, !224, !225, !226, !227, !228, !229, !230, !231}
!222 = !{!"m_OpenCLArgTypesVec[0]", !"float*"}
!223 = !{!"m_OpenCLArgTypesVec[1]", !"class.sycl::_V1::range"}
!224 = !{!"m_OpenCLArgTypesVec[2]", !"class.sycl::_V1::range"}
!225 = !{!"m_OpenCLArgTypesVec[3]", !"long"}
!226 = !{!"m_OpenCLArgTypesVec[4]", !"class.sycl::_V1::ext::oneapi::bfloat16*"}
!227 = !{!"m_OpenCLArgTypesVec[5]", !"class.sycl::_V1::range"}
!228 = !{!"m_OpenCLArgTypesVec[6]", !"class.sycl::_V1::range"}
!229 = !{!"m_OpenCLArgTypesVec[7]", !"class.sycl::_V1::ext::oneapi::bfloat16*"}
!230 = !{!"m_OpenCLArgTypesVec[8]", !"class.sycl::_V1::range"}
!231 = !{!"m_OpenCLArgTypesVec[9]", !"class.sycl::_V1::range"}
!232 = !{!"m_OpenCLArgBaseTypes", !233, !234, !235, !236, !237, !238, !239, !240, !241, !242}
!233 = !{!"m_OpenCLArgBaseTypesVec[0]", !"float*"}
!234 = !{!"m_OpenCLArgBaseTypesVec[1]", !"class.sycl::_V1::range"}
!235 = !{!"m_OpenCLArgBaseTypesVec[2]", !"class.sycl::_V1::range"}
!236 = !{!"m_OpenCLArgBaseTypesVec[3]", !"long"}
!237 = !{!"m_OpenCLArgBaseTypesVec[4]", !"class.sycl::_V1::ext::oneapi::bfloat16*"}
!238 = !{!"m_OpenCLArgBaseTypesVec[5]", !"class.sycl::_V1::range"}
!239 = !{!"m_OpenCLArgBaseTypesVec[6]", !"class.sycl::_V1::range"}
!240 = !{!"m_OpenCLArgBaseTypesVec[7]", !"class.sycl::_V1::ext::oneapi::bfloat16*"}
!241 = !{!"m_OpenCLArgBaseTypesVec[8]", !"class.sycl::_V1::range"}
!242 = !{!"m_OpenCLArgBaseTypesVec[9]", !"class.sycl::_V1::range"}
!243 = !{!"m_OpenCLArgTypeQualifiers", !244, !245, !246, !247, !248, !249, !250, !251, !252, !253}
!244 = !{!"m_OpenCLArgTypeQualifiersVec[0]", !""}
!245 = !{!"m_OpenCLArgTypeQualifiersVec[1]", !""}
!246 = !{!"m_OpenCLArgTypeQualifiersVec[2]", !""}
!247 = !{!"m_OpenCLArgTypeQualifiersVec[3]", !""}
!248 = !{!"m_OpenCLArgTypeQualifiersVec[4]", !""}
!249 = !{!"m_OpenCLArgTypeQualifiersVec[5]", !""}
!250 = !{!"m_OpenCLArgTypeQualifiersVec[6]", !""}
!251 = !{!"m_OpenCLArgTypeQualifiersVec[7]", !""}
!252 = !{!"m_OpenCLArgTypeQualifiersVec[8]", !""}
!253 = !{!"m_OpenCLArgTypeQualifiersVec[9]", !""}
!254 = !{!"m_OpenCLArgNames", !255, !256, !257, !258, !259, !260, !261, !262, !263, !264}
!255 = !{!"m_OpenCLArgNamesVec[0]", !""}
!256 = !{!"m_OpenCLArgNamesVec[1]", !""}
!257 = !{!"m_OpenCLArgNamesVec[2]", !""}
!258 = !{!"m_OpenCLArgNamesVec[3]", !""}
!259 = !{!"m_OpenCLArgNamesVec[4]", !""}
!260 = !{!"m_OpenCLArgNamesVec[5]", !""}
!261 = !{!"m_OpenCLArgNamesVec[6]", !""}
!262 = !{!"m_OpenCLArgNamesVec[7]", !""}
!263 = !{!"m_OpenCLArgNamesVec[8]", !""}
!264 = !{!"m_OpenCLArgNamesVec[9]", !""}
!265 = !{!"m_OpenCLArgScalarAsPointers"}
!266 = !{!"m_OptsToDisablePerFunc"}
!267 = !{!"pushInfo", !268, !269, !270, !274, !275, !276, !277, !278, !279, !280, !281, !294, !295, !296, !297}
!268 = !{!"pushableAddresses"}
!269 = !{!"bindlessPushInfo"}
!270 = !{!"dynamicBufferInfo", !271, !272, !273}
!271 = !{!"firstIndex", i32 0}
!272 = !{!"numOffsets", i32 0}
!273 = !{!"forceDisabled", i1 false}
!274 = !{!"MaxNumberOfPushedBuffers", i32 0}
!275 = !{!"inlineConstantBufferSlot", i32 -1}
!276 = !{!"inlineConstantBufferOffset", i32 -1}
!277 = !{!"inlineConstantBufferGRFOffset", i32 -1}
!278 = !{!"constants"}
!279 = !{!"inputs"}
!280 = !{!"constantReg"}
!281 = !{!"simplePushInfoArr", !282, !291, !292, !293}
!282 = !{!"simplePushInfoArrVec[0]", !283, !284, !285, !286, !287, !288, !289, !290}
!283 = !{!"cbIdx", i32 0}
!284 = !{!"pushableAddressGrfOffset", i32 -1}
!285 = !{!"pushableOffsetGrfOffset", i32 -1}
!286 = !{!"offset", i32 0}
!287 = !{!"size", i32 0}
!288 = !{!"isStateless", i1 false}
!289 = !{!"isBindless", i1 false}
!290 = !{!"simplePushLoads"}
!291 = !{!"simplePushInfoArrVec[1]", !283, !284, !285, !286, !287, !288, !289, !290}
!292 = !{!"simplePushInfoArrVec[2]", !283, !284, !285, !286, !287, !288, !289, !290}
!293 = !{!"simplePushInfoArrVec[3]", !283, !284, !285, !286, !287, !288, !289, !290}
!294 = !{!"simplePushBufferUsed", i32 0}
!295 = !{!"pushAnalysisWIInfos"}
!296 = !{!"inlineRTGlobalPtrOffset", i32 0}
!297 = !{!"rtSyncSurfPtrOffset", i32 0}
!298 = !{!"WaEnableICBPromotion", i1 false}
!299 = !{!"vsInfo", !300, !301, !302}
!300 = !{!"DrawIndirectBufferIndex", i32 -1}
!301 = !{!"vertexReordering", i32 -1}
!302 = !{!"MaxNumOfOutputs", i32 0}
!303 = !{!"hsInfo", !304, !305}
!304 = !{!"numPatchAttributesPatchBaseName", !""}
!305 = !{!"numVertexAttributesPatchBaseName", !""}
!306 = !{!"dsInfo", !302}
!307 = !{!"gsInfo", !302}
!308 = !{!"psInfo", !309, !310, !311, !312, !313, !314, !315, !316, !317, !318, !319, !320, !321, !322, !323, !324, !325, !326, !327, !328, !329, !330, !331, !332, !333, !334, !335, !336, !337, !338, !339, !340, !341, !342}
!309 = !{!"BlendStateDisabledMask", i8 0}
!310 = !{!"SkipSrc0Alpha", i1 false}
!311 = !{!"DualSourceBlendingDisabled", i1 false}
!312 = !{!"ForceEnableSimd32", i1 false}
!313 = !{!"outputDepth", i1 false}
!314 = !{!"outputStencil", i1 false}
!315 = !{!"outputMask", i1 false}
!316 = !{!"blendToFillEnabled", i1 false}
!317 = !{!"forceEarlyZ", i1 false}
!318 = !{!"hasVersionedLoop", i1 false}
!319 = !{!"forceSingleSourceRTWAfterDualSourceRTW", i1 false}
!320 = !{!"requestCPSizeRelevant", i1 false}
!321 = !{!"requestCPSize", i1 false}
!322 = !{!"texelMaskFastClearMode", !"Disabled"}
!323 = !{!"NumSamples", i8 0}
!324 = !{!"blendOptimizationMode"}
!325 = !{!"colorOutputMask"}
!326 = !{!"ProvokingVertexModeNosIndex", i32 0}
!327 = !{!"ProvokingVertexModeNosPatch", !""}
!328 = !{!"ProvokingVertexModeLast", !"Negative"}
!329 = !{!"VertexAttributesBypass", i1 false}
!330 = !{!"LegacyBaryAssignmentDisableLinear", i1 false}
!331 = !{!"LegacyBaryAssignmentDisableLinearNoPerspective", i1 false}
!332 = !{!"LegacyBaryAssignmentDisableLinearCentroid", i1 false}
!333 = !{!"LegacyBaryAssignmentDisableLinearNoPerspectiveCentroid", i1 false}
!334 = !{!"LegacyBaryAssignmentDisableLinearSample", i1 false}
!335 = !{!"LegacyBaryAssignmentDisableLinearNoPerspectiveSample", i1 false}
!336 = !{!"MeshShaderWAPerPrimitiveUserDataEnable", !"Negative"}
!337 = !{!"meshShaderWAPerPrimitiveUserDataEnablePatchName", !""}
!338 = !{!"generatePatchesForRTWriteSends", i1 false}
!339 = !{!"forceVMask", i1 false}
!340 = !{!"WaDisableVRS", i1 false}
!341 = !{!"RelaxMemoryVisibilityFromPSOrdering", i1 false}
!342 = !{!"WaEnableVMaskUnderNonUnifromCF", i1 false}
!343 = !{!"csInfo", !344, !345, !346, !347, !348, !56, !57, !349, !350, !351, !352, !353, !354, !355, !356, !357, !358, !359, !360, !361, !89, !362, !363, !364, !365, !366, !367, !368}
!344 = !{!"maxWorkGroupSize", i32 0}
!345 = !{!"waveSize", i32 0}
!346 = !{!"ComputeShaderSecondCompile"}
!347 = !{!"forcedSIMDSize", i8 0}
!348 = !{!"forceTotalGRFNum", i32 0}
!349 = !{!"forceSpillCompression", i1 false}
!350 = !{!"allowLowerSimd", i1 false}
!351 = !{!"disableSimd32Slicing", i1 false}
!352 = !{!"disableSplitOnSpill", i1 false}
!353 = !{!"enableNewSpillCostFunction", i1 false}
!354 = !{!"forceVISAPreSched", i1 false}
!355 = !{!"forceUniformBuffer", i1 false}
!356 = !{!"forceUniformSurfaceSampler", i1 false}
!357 = !{!"disableLocalIdOrderOptimizations", i1 false}
!358 = !{!"disableDispatchAlongY", i1 false}
!359 = !{!"neededThreadIdLayout", i1* null}
!360 = !{!"forceTileYWalk", i1 false}
!361 = !{!"atomicBranch", i32 0}
!362 = !{!"disableEarlyOut", i1 false}
!363 = !{!"walkOrderEnabled", i1 false}
!364 = !{!"walkOrderOverride", i32 0}
!365 = !{!"ResForHfPacking"}
!366 = !{!"hasWaveMatrix", i1 false}
!367 = !{!"constantFoldSimdSize", i1 false}
!368 = !{!"isNodeShader", i1 false}
!369 = !{!"msInfo", !370, !371, !372, !373, !374, !375, !376, !377, !378, !379, !380, !328, !326, !381}
!370 = !{!"PrimitiveTopology", i32 3}
!371 = !{!"MaxNumOfPrimitives", i32 0}
!372 = !{!"MaxNumOfVertices", i32 0}
!373 = !{!"MaxNumOfPerPrimitiveOutputs", i32 0}
!374 = !{!"MaxNumOfPerVertexOutputs", i32 0}
!375 = !{!"WorkGroupSize", i32 0}
!376 = !{!"WorkGroupMemorySizeInBytes", i32 0}
!377 = !{!"IndexFormat", i32 6}
!378 = !{!"SubgroupSize", i32 0}
!379 = !{!"VPandRTAIndexAutostripEnable", i1 false}
!380 = !{!"MeshShaderWAPerPrimitiveUserDataEnable", i1 false}
!381 = !{!"numPrimitiveAttributesPatchBaseName", !""}
!382 = !{!"taskInfo", !302, !375, !376, !378}
!383 = !{!"NBarrierCnt", i32 0}
!384 = !{!"rtInfo", !385, !386, !387, !388, !389, !390, !391, !392, !393, !394, !395, !396, !397, !398}
!385 = !{!"RayQueryAllocSizeInBytes", i32 0}
!386 = !{!"NumContinuations", i32 0}
!387 = !{!"RTAsyncStackAddrspace", i32 -1}
!388 = !{!"RTAsyncStackSurfaceStateOffset", i1* null}
!389 = !{!"SWHotZoneAddrspace", i32 -1}
!390 = !{!"SWHotZoneSurfaceStateOffset", i1* null}
!391 = !{!"SWStackAddrspace", i32 -1}
!392 = !{!"SWStackSurfaceStateOffset", i1* null}
!393 = !{!"RTSyncStackAddrspace", i32 -1}
!394 = !{!"RTSyncStackSurfaceStateOffset", i1* null}
!395 = !{!"doSyncDispatchRays", i1 false}
!396 = !{!"MemStyle", !"Xe"}
!397 = !{!"GlobalDataStyle", !"Xe"}
!398 = !{!"NeedsBTD", i1 true}
!399 = !{!"EnableTextureIndirection", i1 false}
!400 = !{!"EnableSamplerIndirection", i1 false}
!401 = !{!"samplerStateStride", i32 0}
!402 = !{!"samplerStateOffset", i32 0}
!403 = !{!"textureStateStride", i32 0}
!404 = !{!"textureStateOffset", i32 0}
!405 = !{!"CurUniqueIndirectIdx", i32 0}
!406 = !{!"inlineDynTextures"}
!407 = !{!"inlineResInfoData"}
!408 = !{!"immConstant", !409, !410, !411}
!409 = !{!"data"}
!410 = !{!"sizes"}
!411 = !{!"zeroIdxs"}
!412 = !{!"stringConstants"}
!413 = !{!"inlineBuffers", !414, !418, !419}
!414 = !{!"inlineBuffersVec[0]", !415, !416, !417}
!415 = !{!"alignment", i32 0}
!416 = !{!"allocSize", i64 0}
!417 = !{!"Buffer"}
!418 = !{!"inlineBuffersVec[1]", !415, !416, !417}
!419 = !{!"inlineBuffersVec[2]", !415, !416, !417}
!420 = !{!"GlobalPointerProgramBinaryInfos"}
!421 = !{!"ConstantPointerProgramBinaryInfos"}
!422 = !{!"GlobalBufferAddressRelocInfo"}
!423 = !{!"ConstantBufferAddressRelocInfo"}
!424 = !{!"forceLscCacheList"}
!425 = !{!"SrvMap"}
!426 = !{!"RootConstantBufferOffsetInBytes"}
!427 = !{!"RasterizerOrderedByteAddressBuffer"}
!428 = !{!"RasterizerOrderedViews"}
!429 = !{!"MinNOSPushConstantSize", i32 0}
!430 = !{!"inlineProgramScopeOffsets"}
!431 = !{!"shaderData", !432}
!432 = !{!"numReplicas", i32 0}
!433 = !{!"URBInfo", !434, !435, !436}
!434 = !{!"has64BVertexHeaderInput", i1 false}
!435 = !{!"has64BVertexHeaderOutput", i1 false}
!436 = !{!"hasVertexHeader", i1 true}
!437 = !{!"m_ForcePullModel", i1 false}
!438 = !{!"UseBindlessImage", i1 false}
!439 = !{!"enableRangeReduce", i1 false}
!440 = !{!"disableNewTrigFuncRangeReduction", i1 false}
!441 = !{!"enableFRemToSRemOpt", i1 false}
!442 = !{!"enableSampleptrToLdmsptrSample0", i1 false}
!443 = !{!"enableSampleLptrToLdmsptrSample0", i1 false}
!444 = !{!"WaForceSIMD32MicropolyRasterize", i1 false}
!445 = !{!"allowMatchMadOptimizationforVS", i1 false}
!446 = !{!"disableMatchMadOptimizationForCS", i1 false}
!447 = !{!"disableMemOptforNegativeOffsetLoads", i1 false}
!448 = !{!"enableThreeWayLoadSpiltOpt", i1 false}
!449 = !{!"statefulResourcesNotAliased", i1 false}
!450 = !{!"disableMixMode", i1 false}
!451 = !{!"genericAccessesResolved", i1 false}
!452 = !{!"disableSeparateSpillPvtScratchSpace", i1 false}
!453 = !{!"disableSeparateScratchWA", i1 false}
!454 = !{!"PrivateMemoryPerFG"}
!455 = !{!"m_OptsToDisable"}
!456 = !{!"capabilities", !457}
!457 = !{!"globalVariableDecorationsINTEL", i1 false}
!458 = !{!"m_ShaderResourceViewMcsMask", !459, !460}
!459 = !{!"m_ShaderResourceViewMcsMaskVec[0]", i64 0}
!460 = !{!"m_ShaderResourceViewMcsMaskVec[1]", i64 0}
!461 = !{!"computedDepthMode", i32 0}
!462 = !{!"isHDCFastClearShader", i1 false}
!463 = !{!"argRegisterReservations", !464}
!464 = !{!"argRegisterReservationsVec[0]", i32 0}
!465 = !{!"SIMD16_SpillThreshold", i8 0}
!466 = !{!"SIMD32_SpillThreshold", i8 0}
!467 = !{!"m_CacheControlOption", !468, !469, !470, !471}
!468 = !{!"LscLoadCacheControlOverride", i8 0}
!469 = !{!"LscStoreCacheControlOverride", i8 0}
!470 = !{!"TgmLoadCacheControlOverride", i8 0}
!471 = !{!"TgmStoreCacheControlOverride", i8 0}
!472 = !{i32 2, i32 0}
!473 = !{!"clang version 14.0.5"}
!474 = !{i32 1, !"wchar_size", i32 4}
!475 = !{!476}
!476 = !{i32 4469}
!477 = !{!478, !478, i64 0}
!478 = !{!"int", !479, i64 0}
!479 = !{!"omnipotent char", !480, i64 0}
!480 = !{!"Simple C/C++ TBAA"}
!481 = !{!476, !482}
!482 = !{i32 4470}
