; ------------------------------------------------
; OCL_asm40d739ce99ad8281_0066_Unify_after_AddImplicitArgs.ll
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
  %48 = call spir_func i32 @__builtin_IB_get_group_id(i32 noundef 1) #4
  %49 = call spir_func i32 @__builtin_IB_get_enqueued_local_size(i32 noundef 1) #4
  %50 = mul i32 %49, %48
  %51 = call spir_func i32 @__builtin_IB_get_local_id_y() #4
  %52 = icmp ult i32 %51, 65536
  call void @llvm.assume(i1 %52) #5
  %53 = add i32 %51, %50
  %54 = call spir_func i32 @__builtin_IB_get_global_offset(i32 noundef 1) #4
  %55 = add i32 %53, %54
  %56 = zext i32 %55 to i64
  %57 = icmp ult i64 %56, 2147483648
  call void @llvm.assume(i1 %57)
  %58 = call spir_func i32 @__builtin_IB_get_group_id(i32 noundef 0) #4
  %59 = call spir_func i32 @__builtin_IB_get_enqueued_local_size(i32 noundef 0) #4
  %60 = mul i32 %59, %58
  %61 = call spir_func i32 @__builtin_IB_get_local_id_x() #4
  %62 = icmp ult i32 %61, 65536
  call void @llvm.assume(i1 %62) #5
  %63 = add i32 %61, %60
  %64 = call spir_func i32 @__builtin_IB_get_global_offset(i32 noundef 0) #4
  %65 = add i32 %63, %64
  %66 = zext i32 %65 to i64
  %67 = icmp ult i64 %66, 2147483648
  call void @llvm.assume(i1 %67)
  %68 = call spir_func i32 @__builtin_IB_get_local_id_y() #4
  %69 = icmp ult i32 %68, 65536
  call void @llvm.assume(i1 %69) #5
  %70 = zext i32 %68 to i64
  %71 = icmp ult i64 %70, 2147483648
  call void @llvm.assume(i1 %71)
  %72 = sub nsw i64 %56, %70, !spirv.Decorations !475
  %73 = call spir_func i32 @__builtin_IB_get_local_id_x() #4
  %74 = icmp ult i32 %73, 65536
  call void @llvm.assume(i1 %74) #5
  %75 = zext i32 %73 to i64
  %76 = icmp ult i64 %75, 2147483648
  call void @llvm.assume(i1 %76)
  %77 = sub nsw i64 %66, %75, !spirv.Decorations !475
  %78 = add i64 %23, %22
  %79 = sub i64 0, %78
  %80 = getelementptr inbounds float, float addrspace(1)* %25, i64 %79
  %81 = shl nsw i64 %72, 11, !spirv.Decorations !475
  %82 = getelementptr inbounds float, float addrspace(1)* %80, i64 %81
  %83 = udiv i64 %77, %3
  %84 = shl i64 %83, 5
  %85 = getelementptr inbounds float, float addrspace(1)* %82, i64 %84
  %86 = bitcast [2 x <64 x float>]* %14 to i8*
  %.ascast.i3 = bitcast float addrspace(1)* %85 to i8 addrspace(1)*
  %87 = call spir_func i32 @__builtin_IB_get_local_size(i32 noundef 0) #4
  %88 = call spir_func i32 @__builtin_IB_get_local_size(i32 noundef 1) #4
  %89 = mul i32 %88, %87
  %90 = call spir_func i32 @__builtin_IB_get_local_size(i32 noundef 2) #4
  %91 = mul i32 %89, %90
  %92 = call spir_func i32 @__builtin_IB_get_simd_size() #6
  %93 = add i32 %92, -1
  %94 = and i32 %93, %91
  %95 = icmp eq i32 %94, 0
  br i1 %95, label %120, label %96

96:                                               ; preds = %10
  %97 = call spir_func i32 @__builtin_IB_get_local_id_z() #4
  %98 = call spir_func i32 @__builtin_IB_get_local_size(i32 noundef 1) #4
  %99 = mul i32 %98, %97
  %100 = call spir_func i32 @__builtin_IB_get_local_id_y() #4
  %101 = add i32 %99, %100
  %102 = call spir_func i32 @__builtin_IB_get_local_size(i32 noundef 0) #4
  %103 = mul i32 %101, %102
  %104 = call spir_func i32 @__builtin_IB_get_local_id_x() #4
  %105 = add i32 %103, %104
  %106 = call spir_func i32 @__builtin_IB_get_simd_size() #6
  %107 = udiv i32 %105, %106
  %108 = call spir_func i32 @__builtin_IB_get_local_size(i32 noundef 0) #4
  %109 = call spir_func i32 @__builtin_IB_get_local_size(i32 noundef 1) #4
  %110 = mul i32 %109, %108
  %111 = call spir_func i32 @__builtin_IB_get_local_size(i32 noundef 2) #4
  %112 = mul i32 %110, %111
  %113 = call spir_func i32 @__builtin_IB_get_simd_size() #6
  %114 = add i32 %112, -1
  %115 = add i32 %114, %113
  %116 = call spir_func i32 @__builtin_IB_get_simd_size() #6
  %117 = udiv i32 %115, %116
  %118 = add i32 %117, -1
  %119 = icmp ult i32 %107, %118
  br i1 %119, label %120, label %_Z18get_sub_group_sizev.exit.i4

120:                                              ; preds = %96, %10
  %121 = call spir_func i32 @__builtin_IB_get_simd_size() #6
  br label %_Z18get_sub_group_sizev.exit.i4

_Z18get_sub_group_sizev.exit.i4:                  ; preds = %120, %96
  %122 = phi i32 [ %121, %120 ], [ %94, %96 ]
  %123 = bitcast i8 addrspace(1)* %.ascast.i3 to i32 addrspace(1)*
  %124 = call spir_func i32 @__builtin_IB_get_simd_id() #6
  %125 = sdiv i32 %122, 32
  %126 = bitcast i8* %86 to i32*
  %127 = freeze i32 %124
  %128 = sdiv i32 %127, 32
  %129 = mul i32 %128, 32
  %130 = sub i32 %127, %129
  %131 = sext i32 %130 to i64
  br label %132

132:                                              ; preds = %143, %_Z18get_sub_group_sizev.exit.i4
  %133 = phi i32 [ 0, %_Z18get_sub_group_sizev.exit.i4 ], [ %147, %143 ]
  %134 = mul nsw i32 %133, %125
  %135 = add nsw i32 %134, %128
  %136 = icmp slt i32 %135, 32
  br i1 %136, label %137, label %143

137:                                              ; preds = %132
  %138 = sext i32 %135 to i64
  %139 = mul nsw i64 %138, 64
  %140 = add nsw i64 %139, %131
  %141 = getelementptr inbounds i32, i32 addrspace(1)* %123, i64 %140
  %142 = load i32, i32 addrspace(1)* %141, align 4, !tbaa !477
  br label %143

143:                                              ; preds = %137, %132
  %144 = phi i32 [ %142, %137 ], [ 0, %132 ]
  %145 = zext i32 %133 to i64
  %146 = getelementptr inbounds i32, i32* %126, i64 %145
  store i32 %144, i32* %146, align 4, !tbaa !477
  %147 = add nuw nsw i32 %133, 1
  %148 = icmp ult i32 %133, 127
  br i1 %148, label %132, label %__builtin_spriv_OpJointMatrixLoadINTEL_Accumulator_RowMajor_32x32_i32_128_global_v8i8_pi32_i32.23.exit

__builtin_spriv_OpJointMatrixLoadINTEL_Accumulator_RowMajor_32x32_i32_128_global_v8i8_pi32_i32.23.exit: ; preds = %143
  %149 = bitcast [2 x <64 x float>]* %14 to <64 x float>*
  %150 = load <64 x float>, <64 x float>* %149, align 256
  %151 = getelementptr <64 x float>, <64 x float>* %149, i32 1
  %152 = load <64 x float>, <64 x float>* %151, align 256
  %153 = insertvalue [2 x <64 x float>] undef, <64 x float> %150, 0
  %154 = insertvalue [2 x <64 x float>] %153, <64 x float> %152, 1
  %155 = add i64 %34, %33
  %156 = sub i64 0, %155
  %157 = getelementptr inbounds %"class.sycl::_V1::ext::oneapi::bfloat16", %"class.sycl::_V1::ext::oneapi::bfloat16" addrspace(1)* %36, i64 %156
  %158 = shl nsw i64 %72, 10, !spirv.Decorations !475
  %159 = getelementptr inbounds %"class.sycl::_V1::ext::oneapi::bfloat16", %"class.sycl::_V1::ext::oneapi::bfloat16" addrspace(1)* %157, i64 %158
  %160 = add i64 %45, %44
  %161 = sub i64 0, %160
  %162 = getelementptr inbounds %"class.sycl::_V1::ext::oneapi::bfloat16", %"class.sycl::_V1::ext::oneapi::bfloat16" addrspace(1)* %47, i64 %161
  %163 = shl i64 %83, 6
  %164 = getelementptr %"class.sycl::_V1::ext::oneapi::bfloat16", %"class.sycl::_V1::ext::oneapi::bfloat16" addrspace(1)* %162, i64 %163
  br label %165

165:                                              ; preds = %__builtin_spriv_OpJointMatrixLoadINTEL_PackedB_PackedB_16x32_i16_32_global_v8i8_pi32_i32.25.exit, %__builtin_spriv_OpJointMatrixLoadINTEL_Accumulator_RowMajor_32x32_i32_128_global_v8i8_pi32_i32.23.exit
  %166 = phi i32 [ 0, %__builtin_spriv_OpJointMatrixLoadINTEL_Accumulator_RowMajor_32x32_i32_128_global_v8i8_pi32_i32.23.exit ], [ %284, %__builtin_spriv_OpJointMatrixLoadINTEL_PackedB_PackedB_16x32_i16_32_global_v8i8_pi32_i32.25.exit ]
  %167 = icmp ult i32 %166, 2
  br i1 %167, label %168, label %285

168:                                              ; preds = %165
  %169 = shl nuw nsw i32 %166, 4, !spirv.Decorations !481
  %170 = zext i32 %169 to i64
  %171 = getelementptr inbounds %"class.sycl::_V1::ext::oneapi::bfloat16", %"class.sycl::_V1::ext::oneapi::bfloat16" addrspace(1)* %159, i64 %170
  %172 = bitcast <32 x i32>* %12 to i8*
  %.ascast.i5 = bitcast %"class.sycl::_V1::ext::oneapi::bfloat16" addrspace(1)* %171 to i8 addrspace(1)*
  %173 = call spir_func i32 @__builtin_IB_get_local_size(i32 noundef 0) #4
  %174 = call spir_func i32 @__builtin_IB_get_local_size(i32 noundef 1) #4
  %175 = mul i32 %174, %173
  %176 = call spir_func i32 @__builtin_IB_get_local_size(i32 noundef 2) #4
  %177 = mul i32 %175, %176
  %178 = call spir_func i32 @__builtin_IB_get_simd_size() #6
  %179 = add i32 %178, -1
  %180 = and i32 %179, %177
  %181 = icmp eq i32 %180, 0
  br i1 %181, label %206, label %182

182:                                              ; preds = %168
  %183 = call spir_func i32 @__builtin_IB_get_local_id_z() #4
  %184 = call spir_func i32 @__builtin_IB_get_local_size(i32 noundef 1) #4
  %185 = mul i32 %184, %183
  %186 = call spir_func i32 @__builtin_IB_get_local_id_y() #4
  %187 = add i32 %185, %186
  %188 = call spir_func i32 @__builtin_IB_get_local_size(i32 noundef 0) #4
  %189 = mul i32 %187, %188
  %190 = call spir_func i32 @__builtin_IB_get_local_id_x() #4
  %191 = add i32 %189, %190
  %192 = call spir_func i32 @__builtin_IB_get_simd_size() #6
  %193 = udiv i32 %191, %192
  %194 = call spir_func i32 @__builtin_IB_get_local_size(i32 noundef 0) #4
  %195 = call spir_func i32 @__builtin_IB_get_local_size(i32 noundef 1) #4
  %196 = mul i32 %195, %194
  %197 = call spir_func i32 @__builtin_IB_get_local_size(i32 noundef 2) #4
  %198 = mul i32 %196, %197
  %199 = call spir_func i32 @__builtin_IB_get_simd_size() #6
  %200 = add i32 %198, -1
  %201 = add i32 %200, %199
  %202 = call spir_func i32 @__builtin_IB_get_simd_size() #6
  %203 = udiv i32 %201, %202
  %204 = add i32 %203, -1
  %205 = icmp ult i32 %193, %204
  br i1 %205, label %206, label %_Z18get_sub_group_sizev.exit.i6

206:                                              ; preds = %182, %168
  %207 = call spir_func i32 @__builtin_IB_get_simd_size() #6
  br label %_Z18get_sub_group_sizev.exit.i6

_Z18get_sub_group_sizev.exit.i6:                  ; preds = %206, %182
  %208 = bitcast i8* %172 to i32*
  %209 = bitcast i8 addrspace(1)* %.ascast.i5 to i32 addrspace(1)*
  br label %210

210:                                              ; preds = %210, %_Z18get_sub_group_sizev.exit.i6
  %211 = phi i32 [ 0, %_Z18get_sub_group_sizev.exit.i6 ], [ %217, %210 ]
  %212 = zext i32 %211 to i64
  %213 = mul nsw i64 16, %212
  %214 = getelementptr inbounds i32, i32 addrspace(1)* %209, i64 %213
  %215 = call spir_func i32 @__builtin_IB_simd_block_read_1_global(i32 addrspace(1)* noundef %214) #6
  %216 = getelementptr inbounds i32, i32* %208, i64 %212
  store i32 %215, i32* %216, align 4, !tbaa !477
  %217 = add nuw nsw i32 %211, 1
  %218 = icmp ult i32 %211, 31
  br i1 %218, label %210, label %__builtin_spriv_OpJointMatrixLoadINTEL_PackedA_RowMajor_32x16_i16_32_global_v8i8_pi32_i32.24.exit

__builtin_spriv_OpJointMatrixLoadINTEL_PackedA_RowMajor_32x16_i16_32_global_v8i8_pi32_i32.24.exit: ; preds = %210
  %219 = shl nuw nsw i64 %170, 6, !spirv.Decorations !481
  %220 = getelementptr %"class.sycl::_V1::ext::oneapi::bfloat16", %"class.sycl::_V1::ext::oneapi::bfloat16" addrspace(1)* %164, i64 %219
  %221 = bitcast <32 x i32>* %11 to i8*
  %.ascast.i7 = bitcast %"class.sycl::_V1::ext::oneapi::bfloat16" addrspace(1)* %220 to i8 addrspace(1)*
  %222 = call spir_func i32 @__builtin_IB_get_local_size(i32 noundef 0) #4
  %223 = call spir_func i32 @__builtin_IB_get_local_size(i32 noundef 1) #4
  %224 = mul i32 %223, %222
  %225 = call spir_func i32 @__builtin_IB_get_local_size(i32 noundef 2) #4
  %226 = mul i32 %224, %225
  %227 = call spir_func i32 @__builtin_IB_get_simd_size() #6
  %228 = add i32 %227, -1
  %229 = and i32 %228, %226
  %230 = icmp eq i32 %229, 0
  br i1 %230, label %255, label %231

231:                                              ; preds = %__builtin_spriv_OpJointMatrixLoadINTEL_PackedA_RowMajor_32x16_i16_32_global_v8i8_pi32_i32.24.exit
  %232 = call spir_func i32 @__builtin_IB_get_local_id_z() #4
  %233 = call spir_func i32 @__builtin_IB_get_local_size(i32 noundef 1) #4
  %234 = mul i32 %233, %232
  %235 = call spir_func i32 @__builtin_IB_get_local_id_y() #4
  %236 = add i32 %234, %235
  %237 = call spir_func i32 @__builtin_IB_get_local_size(i32 noundef 0) #4
  %238 = mul i32 %236, %237
  %239 = call spir_func i32 @__builtin_IB_get_local_id_x() #4
  %240 = add i32 %238, %239
  %241 = call spir_func i32 @__builtin_IB_get_simd_size() #6
  %242 = udiv i32 %240, %241
  %243 = call spir_func i32 @__builtin_IB_get_local_size(i32 noundef 0) #4
  %244 = call spir_func i32 @__builtin_IB_get_local_size(i32 noundef 1) #4
  %245 = mul i32 %244, %243
  %246 = call spir_func i32 @__builtin_IB_get_local_size(i32 noundef 2) #4
  %247 = mul i32 %245, %246
  %248 = call spir_func i32 @__builtin_IB_get_simd_size() #6
  %249 = add i32 %247, -1
  %250 = add i32 %249, %248
  %251 = call spir_func i32 @__builtin_IB_get_simd_size() #6
  %252 = udiv i32 %250, %251
  %253 = add i32 %252, -1
  %254 = icmp ult i32 %242, %253
  br i1 %254, label %255, label %_Z18get_sub_group_sizev.exit.i8

255:                                              ; preds = %231, %__builtin_spriv_OpJointMatrixLoadINTEL_PackedA_RowMajor_32x16_i16_32_global_v8i8_pi32_i32.24.exit
  %256 = call spir_func i32 @__builtin_IB_get_simd_size() #6
  br label %_Z18get_sub_group_sizev.exit.i8

_Z18get_sub_group_sizev.exit.i8:                  ; preds = %255, %231
  %257 = phi i32 [ %256, %255 ], [ %229, %231 ]
  %258 = bitcast i8 addrspace(1)* %.ascast.i7 to i32 addrspace(1)*
  %259 = call spir_func i32 @__builtin_IB_get_simd_id() #6
  %260 = sdiv i32 %257, 32
  %261 = bitcast i8* %221 to i32*
  %262 = freeze i32 %259
  %263 = sdiv i32 %262, 32
  %264 = mul i32 %263, 32
  %265 = sub i32 %262, %264
  %266 = sext i32 %265 to i64
  br label %267

267:                                              ; preds = %278, %_Z18get_sub_group_sizev.exit.i8
  %268 = phi i32 [ 0, %_Z18get_sub_group_sizev.exit.i8 ], [ %282, %278 ]
  %269 = mul nsw i32 %268, %260
  %270 = add nsw i32 %269, %263
  %271 = icmp slt i32 %270, 8
  br i1 %271, label %272, label %278

272:                                              ; preds = %267
  %273 = sext i32 %270 to i64
  %274 = mul nsw i64 64, %273
  %275 = add nsw i64 %274, %266
  %276 = getelementptr inbounds i32, i32 addrspace(1)* %258, i64 %275
  %277 = load i32, i32 addrspace(1)* %276, align 4, !tbaa !477
  br label %278

278:                                              ; preds = %272, %267
  %279 = phi i32 [ %277, %272 ], [ 0, %267 ]
  %280 = zext i32 %268 to i64
  %281 = getelementptr inbounds i32, i32* %261, i64 %280
  store i32 %279, i32* %281, align 4, !tbaa !477
  %282 = add nuw nsw i32 %268, 1
  %283 = icmp ult i32 %268, 31
  br i1 %283, label %267, label %__builtin_spriv_OpJointMatrixLoadINTEL_PackedB_PackedB_16x32_i16_32_global_v8i8_pi32_i32.25.exit

__builtin_spriv_OpJointMatrixLoadINTEL_PackedB_PackedB_16x32_i16_32_global_v8i8_pi32_i32.25.exit: ; preds = %278
  %284 = add nuw nsw i32 %166, 1, !spirv.Decorations !481
  br label %165

285:                                              ; preds = %165
  %.fca.0.extract = extractvalue [2 x <64 x float>] %154, 0
  %.fca.0.gep = getelementptr inbounds [2 x <64 x float>], [2 x <64 x float>]* %13, i32 0, i32 0
  store <64 x float> %.fca.0.extract, <64 x float>* %.fca.0.gep, align 256
  %.fca.1.extract = extractvalue [2 x <64 x float>] %154, 1
  %.fca.1.gep = getelementptr inbounds [2 x <64 x float>], [2 x <64 x float>]* %13, i32 0, i32 1
  store <64 x float> %.fca.1.extract, <64 x float>* %.fca.1.gep, align 256
  %286 = bitcast [2 x <64 x float>]* %13 to i8*
  %.ascast.i = bitcast float addrspace(1)* %85 to i8 addrspace(1)*
  %287 = call spir_func i32 @__builtin_IB_get_local_size(i32 noundef 0) #4
  %288 = call spir_func i32 @__builtin_IB_get_local_size(i32 noundef 1) #4
  %289 = mul i32 %288, %287
  %290 = call spir_func i32 @__builtin_IB_get_local_size(i32 noundef 2) #4
  %291 = mul i32 %289, %290
  %292 = call spir_func i32 @__builtin_IB_get_simd_size() #6
  %293 = add i32 %292, -1
  %294 = and i32 %293, %291
  %295 = icmp eq i32 %294, 0
  br i1 %295, label %320, label %296

296:                                              ; preds = %285
  %297 = call spir_func i32 @__builtin_IB_get_local_id_z() #4
  %298 = call spir_func i32 @__builtin_IB_get_local_size(i32 noundef 1) #4
  %299 = mul i32 %298, %297
  %300 = call spir_func i32 @__builtin_IB_get_local_id_y() #4
  %301 = add i32 %299, %300
  %302 = call spir_func i32 @__builtin_IB_get_local_size(i32 noundef 0) #4
  %303 = mul i32 %301, %302
  %304 = call spir_func i32 @__builtin_IB_get_local_id_x() #4
  %305 = add i32 %303, %304
  %306 = call spir_func i32 @__builtin_IB_get_simd_size() #6
  %307 = udiv i32 %305, %306
  %308 = call spir_func i32 @__builtin_IB_get_local_size(i32 noundef 0) #4
  %309 = call spir_func i32 @__builtin_IB_get_local_size(i32 noundef 1) #4
  %310 = mul i32 %309, %308
  %311 = call spir_func i32 @__builtin_IB_get_local_size(i32 noundef 2) #4
  %312 = mul i32 %310, %311
  %313 = call spir_func i32 @__builtin_IB_get_simd_size() #6
  %314 = add i32 %312, -1
  %315 = add i32 %314, %313
  %316 = call spir_func i32 @__builtin_IB_get_simd_size() #6
  %317 = udiv i32 %315, %316
  %318 = add i32 %317, -1
  %319 = icmp ult i32 %307, %318
  br i1 %319, label %320, label %_Z18get_sub_group_sizev.exit.i

320:                                              ; preds = %296, %285
  %321 = call spir_func i32 @__builtin_IB_get_simd_size() #6
  br label %_Z18get_sub_group_sizev.exit.i

_Z18get_sub_group_sizev.exit.i:                   ; preds = %320, %296
  %322 = phi i32 [ %321, %320 ], [ %294, %296 ]
  %323 = bitcast i8 addrspace(1)* %.ascast.i to i32 addrspace(1)*
  %324 = call spir_func i32 @__builtin_IB_get_simd_id() #6
  %325 = sdiv i32 %322, 32
  %326 = bitcast i8* %286 to i32*
  %327 = freeze i32 %324
  %328 = sdiv i32 %327, 32
  %329 = mul i32 %328, 32
  %330 = sub i32 %327, %329
  %331 = sext i32 %330 to i64
  br label %332

332:                                              ; preds = %345, %_Z18get_sub_group_sizev.exit.i
  %333 = phi i32 [ 0, %_Z18get_sub_group_sizev.exit.i ], [ %346, %345 ]
  %334 = mul nsw i32 %333, %325
  %335 = add nsw i32 %334, %328
  %336 = icmp slt i32 %335, 32
  br i1 %336, label %337, label %345

337:                                              ; preds = %332
  %338 = zext i32 %333 to i64
  %339 = getelementptr inbounds i32, i32* %326, i64 %338
  %340 = load i32, i32* %339, align 4, !tbaa !477
  %341 = sext i32 %335 to i64
  %342 = mul nsw i64 %341, 64
  %343 = add nsw i64 %342, %331
  %344 = getelementptr inbounds i32, i32 addrspace(1)* %323, i64 %343
  store i32 %340, i32 addrspace(1)* %344, align 4, !tbaa !477
  br label %345

345:                                              ; preds = %337, %332
  %346 = add nuw nsw i32 %333, 1
  %347 = icmp ult i32 %333, 127
  br i1 %347, label %332, label %__builtin_spriv_OpJointMatrixStoreINTEL_Accumulator_RowMajor_32x32_i32_128_global_pi64_v8i8.26.exit

__builtin_spriv_OpJointMatrixStoreINTEL_Accumulator_RowMajor_32x32_i32_128_global_pi64_v8i8.26.exit: ; preds = %345
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
attributes #4 = { convergent nounwind readnone willreturn }
attributes #5 = { nounwind }
attributes #6 = { convergent nounwind }

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
