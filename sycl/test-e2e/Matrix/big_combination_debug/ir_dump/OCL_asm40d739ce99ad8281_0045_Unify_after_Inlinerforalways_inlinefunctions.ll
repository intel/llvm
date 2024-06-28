; ------------------------------------------------
; OCL_asm40d739ce99ad8281_0045_Unify_after_Inlinerforalways_inlinefunctions.ll
; LLVM major version: 14
; ------------------------------------------------

Printing <null> Function
; Function Attrs: convergent nounwind
define spir_kernel void @_ZTS7imatrixIfLm32ELm32ELm16EE(float addrspace(1)* align 4 %0, %"class.sycl::_V1::range"* byval(%"class.sycl::_V1::range") align 8 %1, %"class.sycl::_V1::range"* byval(%"class.sycl::_V1::range") align 8 %2, i64 %3, %"class.sycl::_V1::ext::oneapi::bfloat16" addrspace(1)* align 2 %4, %"class.sycl::_V1::range"* byval(%"class.sycl::_V1::range") align 8 %5, %"class.sycl::_V1::range"* byval(%"class.sycl::_V1::range") align 8 %6, %"class.sycl::_V1::ext::oneapi::bfloat16" addrspace(1)* align 2 %7, %"class.sycl::_V1::range"* byval(%"class.sycl::_V1::range") align 8 %8, %"class.sycl::_V1::range"* byval(%"class.sycl::_V1::range") align 8 %9) #0 !kernel_arg_addr_space !440 !kernel_arg_access_qual !441 !kernel_arg_type !442 !kernel_arg_type_qual !443 !kernel_arg_base_type !442 !kernel_arg_name !443 !spirv.ParameterDecorations !444 {
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
  %72 = sub nsw i64 %56, %70, !spirv.Decorations !453
  %73 = call spir_func i32 @__builtin_IB_get_local_id_x() #4
  %74 = icmp ult i32 %73, 65536
  call void @llvm.assume(i1 %74) #5
  %75 = zext i32 %73 to i64
  %76 = icmp ult i64 %75, 2147483648
  call void @llvm.assume(i1 %76)
  %77 = sub nsw i64 %66, %75, !spirv.Decorations !453
  %78 = add i64 %23, %22
  %79 = sub i64 0, %78
  %80 = getelementptr inbounds float, float addrspace(1)* %25, i64 %79
  %81 = shl nsw i64 %72, 11, !spirv.Decorations !453
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
  %142 = load i32, i32 addrspace(1)* %141, align 4, !tbaa !455
  br label %143

143:                                              ; preds = %137, %132
  %144 = phi i32 [ %142, %137 ], [ 0, %132 ]
  %145 = zext i32 %133 to i64
  %146 = getelementptr inbounds i32, i32* %126, i64 %145
  store i32 %144, i32* %146, align 4, !tbaa !455
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
  %158 = shl nsw i64 %72, 10, !spirv.Decorations !453
  %159 = getelementptr inbounds %"class.sycl::_V1::ext::oneapi::bfloat16", %"class.sycl::_V1::ext::oneapi::bfloat16" addrspace(1)* %157, i64 %158
  %160 = add i64 %45, %44
  %161 = sub i64 0, %160
  %162 = getelementptr inbounds %"class.sycl::_V1::ext::oneapi::bfloat16", %"class.sycl::_V1::ext::oneapi::bfloat16" addrspace(1)* %47, i64 %161
  %163 = shl i64 %83, 6
  %164 = getelementptr %"class.sycl::_V1::ext::oneapi::bfloat16", %"class.sycl::_V1::ext::oneapi::bfloat16" addrspace(1)* %162, i64 %163
  br label %165

165:                                              ; preds = %__builtin_spriv_OpJointMatrixLoadINTEL_PackedB_PackedB_16x32_i16_32_global_v8i8_pi32_i32.25.exit, %__builtin_spriv_OpJointMatrixLoadINTEL_Accumulator_RowMajor_32x32_i32_128_global_v8i8_pi32_i32.23.exit
  %166 = phi i32 [ 0, %__builtin_spriv_OpJointMatrixLoadINTEL_Accumulator_RowMajor_32x32_i32_128_global_v8i8_pi32_i32.23.exit ], [ %287, %__builtin_spriv_OpJointMatrixLoadINTEL_PackedB_PackedB_16x32_i16_32_global_v8i8_pi32_i32.25.exit ]
  %167 = icmp ult i32 %166, 2
  br i1 %167, label %168, label %288

168:                                              ; preds = %165
  %169 = shl nuw nsw i32 %166, 4, !spirv.Decorations !459
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
  %208 = phi i32 [ %207, %206 ], [ %180, %182 ]
  %209 = bitcast i8* %172 to i32*
  %210 = bitcast i8 addrspace(1)* %.ascast.i5 to i32 addrspace(1)*
  br label %211

211:                                              ; preds = %211, %_Z18get_sub_group_sizev.exit.i6
  %212 = phi i32 [ 0, %_Z18get_sub_group_sizev.exit.i6 ], [ %218, %211 ]
  %213 = zext i32 %212 to i64
  %214 = mul nsw i64 16, %213
  %215 = getelementptr inbounds i32, i32 addrspace(1)* %210, i64 %214
  %216 = call spir_func i32 @__builtin_IB_simd_block_read_1_global(i32 addrspace(1)* noundef %215) #6
  %217 = getelementptr inbounds i32, i32* %209, i64 %213
  store i32 %216, i32* %217, align 4, !tbaa !455
  %218 = add nuw nsw i32 %212, 1
  %219 = icmp ult i32 %212, 31
  br i1 %219, label %211, label %__builtin_spriv_OpJointMatrixLoadINTEL_PackedA_RowMajor_32x16_i16_32_global_v8i8_pi32_i32.24.exit

__builtin_spriv_OpJointMatrixLoadINTEL_PackedA_RowMajor_32x16_i16_32_global_v8i8_pi32_i32.24.exit: ; preds = %211
  %220 = load <32 x i32>, <32 x i32>* %12, align 128
  %221 = shl nuw nsw i64 %170, 6, !spirv.Decorations !459
  %222 = getelementptr %"class.sycl::_V1::ext::oneapi::bfloat16", %"class.sycl::_V1::ext::oneapi::bfloat16" addrspace(1)* %164, i64 %221
  %223 = bitcast <32 x i32>* %11 to i8*
  %.ascast.i7 = bitcast %"class.sycl::_V1::ext::oneapi::bfloat16" addrspace(1)* %222 to i8 addrspace(1)*
  %224 = call spir_func i32 @__builtin_IB_get_local_size(i32 noundef 0) #4
  %225 = call spir_func i32 @__builtin_IB_get_local_size(i32 noundef 1) #4
  %226 = mul i32 %225, %224
  %227 = call spir_func i32 @__builtin_IB_get_local_size(i32 noundef 2) #4
  %228 = mul i32 %226, %227
  %229 = call spir_func i32 @__builtin_IB_get_simd_size() #6
  %230 = add i32 %229, -1
  %231 = and i32 %230, %228
  %232 = icmp eq i32 %231, 0
  br i1 %232, label %257, label %233

233:                                              ; preds = %__builtin_spriv_OpJointMatrixLoadINTEL_PackedA_RowMajor_32x16_i16_32_global_v8i8_pi32_i32.24.exit
  %234 = call spir_func i32 @__builtin_IB_get_local_id_z() #4
  %235 = call spir_func i32 @__builtin_IB_get_local_size(i32 noundef 1) #4
  %236 = mul i32 %235, %234
  %237 = call spir_func i32 @__builtin_IB_get_local_id_y() #4
  %238 = add i32 %236, %237
  %239 = call spir_func i32 @__builtin_IB_get_local_size(i32 noundef 0) #4
  %240 = mul i32 %238, %239
  %241 = call spir_func i32 @__builtin_IB_get_local_id_x() #4
  %242 = add i32 %240, %241
  %243 = call spir_func i32 @__builtin_IB_get_simd_size() #6
  %244 = udiv i32 %242, %243
  %245 = call spir_func i32 @__builtin_IB_get_local_size(i32 noundef 0) #4
  %246 = call spir_func i32 @__builtin_IB_get_local_size(i32 noundef 1) #4
  %247 = mul i32 %246, %245
  %248 = call spir_func i32 @__builtin_IB_get_local_size(i32 noundef 2) #4
  %249 = mul i32 %247, %248
  %250 = call spir_func i32 @__builtin_IB_get_simd_size() #6
  %251 = add i32 %249, -1
  %252 = add i32 %251, %250
  %253 = call spir_func i32 @__builtin_IB_get_simd_size() #6
  %254 = udiv i32 %252, %253
  %255 = add i32 %254, -1
  %256 = icmp ult i32 %244, %255
  br i1 %256, label %257, label %_Z18get_sub_group_sizev.exit.i8

257:                                              ; preds = %233, %__builtin_spriv_OpJointMatrixLoadINTEL_PackedA_RowMajor_32x16_i16_32_global_v8i8_pi32_i32.24.exit
  %258 = call spir_func i32 @__builtin_IB_get_simd_size() #6
  br label %_Z18get_sub_group_sizev.exit.i8

_Z18get_sub_group_sizev.exit.i8:                  ; preds = %257, %233
  %259 = phi i32 [ %258, %257 ], [ %231, %233 ]
  %260 = bitcast i8 addrspace(1)* %.ascast.i7 to i32 addrspace(1)*
  %261 = call spir_func i32 @__builtin_IB_get_simd_id() #6
  %262 = sdiv i32 %259, 32
  %263 = bitcast i8* %223 to i32*
  %264 = freeze i32 %261
  %265 = sdiv i32 %264, 32
  %266 = mul i32 %265, 32
  %267 = sub i32 %264, %266
  %268 = sext i32 %267 to i64
  br label %269

269:                                              ; preds = %280, %_Z18get_sub_group_sizev.exit.i8
  %270 = phi i32 [ 0, %_Z18get_sub_group_sizev.exit.i8 ], [ %284, %280 ]
  %271 = mul nsw i32 %270, %262
  %272 = add nsw i32 %271, %265
  %273 = icmp slt i32 %272, 8
  br i1 %273, label %274, label %280

274:                                              ; preds = %269
  %275 = sext i32 %272 to i64
  %276 = mul nsw i64 64, %275
  %277 = add nsw i64 %276, %268
  %278 = getelementptr inbounds i32, i32 addrspace(1)* %260, i64 %277
  %279 = load i32, i32 addrspace(1)* %278, align 4, !tbaa !455
  br label %280

280:                                              ; preds = %274, %269
  %281 = phi i32 [ %279, %274 ], [ 0, %269 ]
  %282 = zext i32 %270 to i64
  %283 = getelementptr inbounds i32, i32* %263, i64 %282
  store i32 %281, i32* %283, align 4, !tbaa !455
  %284 = add nuw nsw i32 %270, 1
  %285 = icmp ult i32 %270, 31
  br i1 %285, label %269, label %__builtin_spriv_OpJointMatrixLoadINTEL_PackedB_PackedB_16x32_i16_32_global_v8i8_pi32_i32.25.exit

__builtin_spriv_OpJointMatrixLoadINTEL_PackedB_PackedB_16x32_i16_32_global_v8i8_pi32_i32.25.exit: ; preds = %280
  %286 = load <32 x i32>, <32 x i32>* %11, align 128
  %287 = add nuw nsw i32 %166, 1, !spirv.Decorations !459
  br label %165

288:                                              ; preds = %165
  %.fca.0.extract = extractvalue [2 x <64 x float>] %154, 0
  %.fca.0.gep = getelementptr inbounds [2 x <64 x float>], [2 x <64 x float>]* %13, i32 0, i32 0
  store <64 x float> %.fca.0.extract, <64 x float>* %.fca.0.gep, align 256
  %.fca.1.extract = extractvalue [2 x <64 x float>] %154, 1
  %.fca.1.gep = getelementptr inbounds [2 x <64 x float>], [2 x <64 x float>]* %13, i32 0, i32 1
  store <64 x float> %.fca.1.extract, <64 x float>* %.fca.1.gep, align 256
  %289 = bitcast [2 x <64 x float>]* %13 to i8*
  %.ascast.i = bitcast float addrspace(1)* %85 to i8 addrspace(1)*
  %290 = call spir_func i32 @__builtin_IB_get_local_size(i32 noundef 0) #4
  %291 = call spir_func i32 @__builtin_IB_get_local_size(i32 noundef 1) #4
  %292 = mul i32 %291, %290
  %293 = call spir_func i32 @__builtin_IB_get_local_size(i32 noundef 2) #4
  %294 = mul i32 %292, %293
  %295 = call spir_func i32 @__builtin_IB_get_simd_size() #6
  %296 = add i32 %295, -1
  %297 = and i32 %296, %294
  %298 = icmp eq i32 %297, 0
  br i1 %298, label %323, label %299

299:                                              ; preds = %288
  %300 = call spir_func i32 @__builtin_IB_get_local_id_z() #4
  %301 = call spir_func i32 @__builtin_IB_get_local_size(i32 noundef 1) #4
  %302 = mul i32 %301, %300
  %303 = call spir_func i32 @__builtin_IB_get_local_id_y() #4
  %304 = add i32 %302, %303
  %305 = call spir_func i32 @__builtin_IB_get_local_size(i32 noundef 0) #4
  %306 = mul i32 %304, %305
  %307 = call spir_func i32 @__builtin_IB_get_local_id_x() #4
  %308 = add i32 %306, %307
  %309 = call spir_func i32 @__builtin_IB_get_simd_size() #6
  %310 = udiv i32 %308, %309
  %311 = call spir_func i32 @__builtin_IB_get_local_size(i32 noundef 0) #4
  %312 = call spir_func i32 @__builtin_IB_get_local_size(i32 noundef 1) #4
  %313 = mul i32 %312, %311
  %314 = call spir_func i32 @__builtin_IB_get_local_size(i32 noundef 2) #4
  %315 = mul i32 %313, %314
  %316 = call spir_func i32 @__builtin_IB_get_simd_size() #6
  %317 = add i32 %315, -1
  %318 = add i32 %317, %316
  %319 = call spir_func i32 @__builtin_IB_get_simd_size() #6
  %320 = udiv i32 %318, %319
  %321 = add i32 %320, -1
  %322 = icmp ult i32 %310, %321
  br i1 %322, label %323, label %_Z18get_sub_group_sizev.exit.i

323:                                              ; preds = %299, %288
  %324 = call spir_func i32 @__builtin_IB_get_simd_size() #6
  br label %_Z18get_sub_group_sizev.exit.i

_Z18get_sub_group_sizev.exit.i:                   ; preds = %323, %299
  %325 = phi i32 [ %324, %323 ], [ %297, %299 ]
  %326 = bitcast i8 addrspace(1)* %.ascast.i to i32 addrspace(1)*
  %327 = call spir_func i32 @__builtin_IB_get_simd_id() #6
  %328 = sdiv i32 %325, 32
  %329 = bitcast i8* %289 to i32*
  %330 = freeze i32 %327
  %331 = sdiv i32 %330, 32
  %332 = mul i32 %331, 32
  %333 = sub i32 %330, %332
  %334 = sext i32 %333 to i64
  br label %335

335:                                              ; preds = %348, %_Z18get_sub_group_sizev.exit.i
  %336 = phi i32 [ 0, %_Z18get_sub_group_sizev.exit.i ], [ %349, %348 ]
  %337 = mul nsw i32 %336, %328
  %338 = add nsw i32 %337, %331
  %339 = icmp slt i32 %338, 32
  br i1 %339, label %340, label %348

340:                                              ; preds = %335
  %341 = zext i32 %336 to i64
  %342 = getelementptr inbounds i32, i32* %329, i64 %341
  %343 = load i32, i32* %342, align 4, !tbaa !455
  %344 = sext i32 %338 to i64
  %345 = mul nsw i64 %344, 64
  %346 = add nsw i64 %345, %334
  %347 = getelementptr inbounds i32, i32 addrspace(1)* %326, i64 %346
  store i32 %343, i32 addrspace(1)* %347, align 4, !tbaa !455
  br label %348

348:                                              ; preds = %340, %335
  %349 = add nuw nsw i32 %336, 1
  %350 = icmp ult i32 %336, 127
  br i1 %350, label %335, label %__builtin_spriv_OpJointMatrixStoreINTEL_Accumulator_RowMajor_32x32_i32_128_global_pi64_v8i8.26.exit

__builtin_spriv_OpJointMatrixStoreINTEL_Accumulator_RowMajor_32x32_i32_128_global_pi64_v8i8.26.exit: ; preds = %348
  ret void
}

Printing <null> Function
