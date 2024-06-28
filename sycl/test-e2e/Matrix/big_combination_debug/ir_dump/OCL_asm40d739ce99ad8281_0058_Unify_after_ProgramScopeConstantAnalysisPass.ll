; ------------------------------------------------
; OCL_asm40d739ce99ad8281_0058_Unify_after_ProgramScopeConstantAnalysisPass.ll
; LLVM major version: 14
; ------------------------------------------------
target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v16:16:16-v24:32:32-v32:32:32-v48:64:64-v64:64:64-v96:128:128-v128:128:128-v192:256:256-v256:256:256-v512:512:512-v1024:1024:1024-n8:16:32"
target triple = "spir64-unknown-unknown"

%"class.sycl::_V1::ext::oneapi::bfloat16" = type { i16 }
%"class.sycl::_V1::range" = type { %"class.sycl::_V1::detail::array" }
%"class.sycl::_V1::detail::array" = type { [2 x i64] }

; Function Attrs: convergent nounwind
define spir_kernel void @_ZTS7imatrixIfLm32ELm32ELm16EE(float addrspace(1)* align 4 %0, %"class.sycl::_V1::range"* byval(%"class.sycl::_V1::range") align 8 %1, %"class.sycl::_V1::range"* byval(%"class.sycl::_V1::range") align 8 %2, i64 %3, %"class.sycl::_V1::ext::oneapi::bfloat16" addrspace(1)* align 2 %4, %"class.sycl::_V1::range"* byval(%"class.sycl::_V1::range") align 8 %5, %"class.sycl::_V1::range"* byval(%"class.sycl::_V1::range") align 8 %6, %"class.sycl::_V1::ext::oneapi::bfloat16" addrspace(1)* align 2 %7, %"class.sycl::_V1::range"* byval(%"class.sycl::_V1::range") align 8 %8, %"class.sycl::_V1::range"* byval(%"class.sycl::_V1::range") align 8 %9) #0 !kernel_arg_addr_space !448 !kernel_arg_access_qual !449 !kernel_arg_type !450 !kernel_arg_type_qual !451 !kernel_arg_base_type !450 !kernel_arg_name !451 !spirv.ParameterDecorations !452 {
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
  %72 = sub nsw i64 %56, %70, !spirv.Decorations !461
  %73 = call spir_func i32 @__builtin_IB_get_local_id_x() #4
  %74 = icmp ult i32 %73, 65536
  call void @llvm.assume(i1 %74) #5
  %75 = zext i32 %73 to i64
  %76 = icmp ult i64 %75, 2147483648
  call void @llvm.assume(i1 %76)
  %77 = sub nsw i64 %66, %75, !spirv.Decorations !461
  %78 = add i64 %23, %22
  %79 = sub i64 0, %78
  %80 = getelementptr inbounds float, float addrspace(1)* %25, i64 %79
  %81 = shl nsw i64 %72, 11, !spirv.Decorations !461
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
  %142 = load i32, i32 addrspace(1)* %141, align 4, !tbaa !463
  br label %143

143:                                              ; preds = %137, %132
  %144 = phi i32 [ %142, %137 ], [ 0, %132 ]
  %145 = zext i32 %133 to i64
  %146 = getelementptr inbounds i32, i32* %126, i64 %145
  store i32 %144, i32* %146, align 4, !tbaa !463
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
  %158 = shl nsw i64 %72, 10, !spirv.Decorations !461
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
  %169 = shl nuw nsw i32 %166, 4, !spirv.Decorations !467
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
  store i32 %215, i32* %216, align 4, !tbaa !463
  %217 = add nuw nsw i32 %211, 1
  %218 = icmp ult i32 %211, 31
  br i1 %218, label %210, label %__builtin_spriv_OpJointMatrixLoadINTEL_PackedA_RowMajor_32x16_i16_32_global_v8i8_pi32_i32.24.exit

__builtin_spriv_OpJointMatrixLoadINTEL_PackedA_RowMajor_32x16_i16_32_global_v8i8_pi32_i32.24.exit: ; preds = %210
  %219 = shl nuw nsw i64 %170, 6, !spirv.Decorations !467
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
  %277 = load i32, i32 addrspace(1)* %276, align 4, !tbaa !463
  br label %278

278:                                              ; preds = %272, %267
  %279 = phi i32 [ %277, %272 ], [ 0, %267 ]
  %280 = zext i32 %268 to i64
  %281 = getelementptr inbounds i32, i32* %261, i64 %280
  store i32 %279, i32* %281, align 4, !tbaa !463
  %282 = add nuw nsw i32 %268, 1
  %283 = icmp ult i32 %268, 31
  br i1 %283, label %267, label %__builtin_spriv_OpJointMatrixLoadINTEL_PackedB_PackedB_16x32_i16_32_global_v8i8_pi32_i32.25.exit

__builtin_spriv_OpJointMatrixLoadINTEL_PackedB_PackedB_16x32_i16_32_global_v8i8_pi32_i32.25.exit: ; preds = %278
  %284 = add nuw nsw i32 %166, 1, !spirv.Decorations !467
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
  %340 = load i32, i32* %339, align 4, !tbaa !463
  %341 = sext i32 %335 to i64
  %342 = mul nsw i64 %341, 64
  %343 = add nsw i64 %342, %331
  %344 = getelementptr inbounds i32, i32 addrspace(1)* %323, i64 %343
  store i32 %340, i32 addrspace(1)* %344, align 4, !tbaa !463
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
!IGCMetadata = !{!15}
!opencl.ocl.version = !{!445, !445, !445, !445, !445}
!opencl.spir.version = !{!445, !445, !445, !445, !445}
!llvm.ident = !{!446, !446, !446, !446, !446}
!llvm.module.flags = !{!447}

!0 = !{i32 2, i32 2}
!1 = !{i32 4, i32 100000}
!2 = !{i16 6, i16 14}
!3 = !{void (float addrspace(1)*, %"class.sycl::_V1::range"*, %"class.sycl::_V1::range"*, i64, %"class.sycl::_V1::ext::oneapi::bfloat16" addrspace(1)*, %"class.sycl::_V1::range"*, %"class.sycl::_V1::range"*, %"class.sycl::_V1::ext::oneapi::bfloat16" addrspace(1)*, %"class.sycl::_V1::range"*, %"class.sycl::_V1::range"*)* @_ZTS7imatrixIfLm32ELm32ELm16EE, !4}
!4 = !{!5, !6, !14}
!5 = !{!"function_type", i32 0}
!6 = !{!"implicit_arg_desc", !7, !8, !9, !10, !11, !12, !13}
!7 = !{i32 0}
!8 = !{i32 1}
!9 = !{i32 5}
!10 = !{i32 7}
!11 = !{i32 8}
!12 = !{i32 9}
!13 = !{i32 6}
!14 = !{!"sub_group_size", i32 8}
!15 = !{!"ModuleMD", !16, !17, !114, !240, !271, !272, !276, !279, !280, !281, !316, !342, !355, !356, !357, !372, !373, !374, !375, !376, !377, !378, !379, !380, !381, !385, !386, !393, !394, !395, !396, !397, !398, !399, !400, !401, !402, !403, !404, !406, !410, !411, !412, !413, !414, !415, !416, !417, !418, !419, !420, !421, !422, !423, !424, !425, !426, !164, !427, !428, !429, !431, !434, !435, !436, !438, !439, !440}
!16 = !{!"isPrecise", i1 false}
!17 = !{!"compOpt", !18, !19, !20, !21, !22, !23, !24, !25, !26, !27, !28, !29, !30, !31, !32, !33, !34, !35, !36, !37, !38, !39, !40, !41, !42, !43, !44, !45, !46, !47, !48, !49, !50, !51, !52, !53, !54, !55, !56, !57, !58, !59, !60, !61, !62, !63, !64, !65, !66, !67, !68, !69, !70, !71, !72, !73, !74, !75, !76, !77, !78, !79, !80, !81, !82, !83, !84, !85, !86, !87, !88, !89, !90, !91, !92, !93, !94, !95, !96, !97, !98, !99, !100, !101, !102, !103, !104, !105, !106, !107, !108, !109, !110, !111, !112, !113}
!18 = !{!"DenormsAreZero", i1 false}
!19 = !{!"BFTFDenormsAreZero", i1 false}
!20 = !{!"CorrectlyRoundedDivSqrt", i1 false}
!21 = !{!"OptDisable", i1 false}
!22 = !{!"MadEnable", i1 true}
!23 = !{!"NoSignedZeros", i1 false}
!24 = !{!"NoNaNs", i1 false}
!25 = !{!"FloatRoundingMode", i32 0}
!26 = !{!"FloatCvtIntRoundingMode", i32 3}
!27 = !{!"LoadCacheDefault", i32 4}
!28 = !{!"StoreCacheDefault", i32 7}
!29 = !{!"VISAPreSchedRPThreshold", i32 0}
!30 = !{!"SetLoopUnrollThreshold", i32 0}
!31 = !{!"UnsafeMathOptimizations", i1 false}
!32 = !{!"disableCustomUnsafeOpts", i1 false}
!33 = !{!"disableReducePow", i1 false}
!34 = !{!"disableSqrtOpt", i1 false}
!35 = !{!"FiniteMathOnly", i1 false}
!36 = !{!"FastRelaxedMath", i1 false}
!37 = !{!"DashGSpecified", i1 false}
!38 = !{!"FastCompilation", i1 false}
!39 = !{!"UseScratchSpacePrivateMemory", i1 true}
!40 = !{!"RelaxedBuiltins", i1 false}
!41 = !{!"SubgroupIndependentForwardProgressRequired", i1 true}
!42 = !{!"GreaterThan2GBBufferRequired", i1 true}
!43 = !{!"GreaterThan4GBBufferRequired", i1 false}
!44 = !{!"DisableA64WA", i1 false}
!45 = !{!"ForceEnableA64WA", i1 false}
!46 = !{!"PushConstantsEnable", i1 true}
!47 = !{!"HasPositivePointerOffset", i1 false}
!48 = !{!"HasBufferOffsetArg", i1 true}
!49 = !{!"BufferOffsetArgOptional", i1 true}
!50 = !{!"replaceGlobalOffsetsByZero", i1 false}
!51 = !{!"forcePixelShaderSIMDMode", i32 0}
!52 = !{!"ForceGeomFFShaderSIMDMode", i32 0}
!53 = !{!"pixelShaderDoNotAbortOnSpill", i1 false}
!54 = !{!"UniformWGS", i1 false}
!55 = !{!"disableVertexComponentPacking", i1 false}
!56 = !{!"disablePartialVertexComponentPacking", i1 false}
!57 = !{!"PreferBindlessImages", i1 false}
!58 = !{!"UseBindlessMode", i1 false}
!59 = !{!"UseLegacyBindlessMode", i1 true}
!60 = !{!"disableMathRefactoring", i1 false}
!61 = !{!"atomicBranch", i1 false}
!62 = !{!"spillCompression", i1 false}
!63 = !{!"DisableEarlyOut", i1 false}
!64 = !{!"ForceInt32DivRemEmu", i1 false}
!65 = !{!"ForceInt32DivRemEmuSP", i1 false}
!66 = !{!"WaveIntrinsicUsed", i1 false}
!67 = !{!"DisableMultiPolyPS", i1 false}
!68 = !{!"NeedTexture3DLODWA", i1 false}
!69 = !{!"DisableFastestSingleCSSIMD", i1 false}
!70 = !{!"DisableFastestLinearScan", i1 false}
!71 = !{!"UseStatelessforPrivateMemory", i1 false}
!72 = !{!"EnableTakeGlobalAddress", i1 false}
!73 = !{!"IsLibraryCompilation", i1 false}
!74 = !{!"LibraryCompileSIMDSize", i32 0}
!75 = !{!"FastVISACompile", i1 false}
!76 = !{!"MatchSinCosPi", i1 false}
!77 = !{!"ExcludeIRFromZEBinary", i1 false}
!78 = !{!"EmitZeBinVISASections", i1 false}
!79 = !{!"FP64GenEmulationEnabled", i1 false}
!80 = !{!"FP64GenConvEmulationEnabled", i1 false}
!81 = !{!"allowDisableRematforCS", i1 false}
!82 = !{!"DisableIncSpillCostAllAddrTaken", i1 false}
!83 = !{!"DisableCPSOmaskWA", i1 false}
!84 = !{!"DisableFastestGopt", i1 false}
!85 = !{!"WaForceHalfPromotionComputeShader", i1 false}
!86 = !{!"WaForceHalfPromotionPixelVertexShader", i1 false}
!87 = !{!"DisableConstantCoalescing", i1 false}
!88 = !{!"EnableUndefAlphaOutputAsRed", i1 true}
!89 = !{!"WaEnableALTModeVisaWA", i1 false}
!90 = !{!"WaEnableAtomicWaveFusion", i1 false}
!91 = !{!"WaEnableAtomicWaveFusionNonNullResource", i1 false}
!92 = !{!"WaEnableAtomicWaveFusionStateless", i1 false}
!93 = !{!"WaEnableAtomicWaveFusionTyped", i1 false}
!94 = !{!"ForceCBThroughSampler3D", i1 false}
!95 = !{!"WaStoreRawVectorToTypedWrite", i1 false}
!96 = !{!"WaLoadRawVectorToTypedRead", i1 false}
!97 = !{!"WaZeroSLMBeforeUse", i1 false}
!98 = !{!"WaFlagGroupTypedUAVGloballyCoherent", i1 false}
!99 = !{!"NewSpillCostFunction", i1 false}
!100 = !{!"EnableVRT", i1 false}
!101 = !{!"ForceLargeGRFNum4RQ", i1 false}
!102 = !{!"Enable2xGRFRetry", i1 false}
!103 = !{!"Detect2xGRFCandidate", i1 false}
!104 = !{!"EnableURBWritesMerging", i1 true}
!105 = !{!"DisableEUFusion", i1 false}
!106 = !{!"DisableFDivToFMulInvOpt", i1 false}
!107 = !{!"initializePhiSampleSourceWA", i1 false}
!108 = !{!"WaDisableSubspanUseNoMaskForCB", i1 false}
!109 = !{!"DisableLoosenSimd32Occu", i1 false}
!110 = !{!"FastestS1Options", i32 0}
!111 = !{!"DisableFastestForWaveIntrinsicsCS", i1 false}
!112 = !{!"ForceLinearWalkOnLinearUAV", i1 false}
!113 = !{!"DisableLscSamplerRouting", i1 false}
!114 = !{!"FuncMD", !115, !116}
!115 = !{!"FuncMDMap[0]", void (float addrspace(1)*, %"class.sycl::_V1::range"*, %"class.sycl::_V1::range"*, i64, %"class.sycl::_V1::ext::oneapi::bfloat16" addrspace(1)*, %"class.sycl::_V1::range"*, %"class.sycl::_V1::range"*, %"class.sycl::_V1::ext::oneapi::bfloat16" addrspace(1)*, %"class.sycl::_V1::range"*, %"class.sycl::_V1::range"*)* @_ZTS7imatrixIfLm32ELm32ELm16EE}
!116 = !{!"FuncMDValue[0]", !117, !118, !122, !123, !124, !125, !126, !127, !128, !150, !156, !157, !158, !159, !160, !161, !162, !163, !164, !165, !166, !167, !168, !169, !170, !171, !172, !183, !194, !205, !216, !227, !238, !239}
!117 = !{!"localOffsets"}
!118 = !{!"workGroupWalkOrder", !119, !120, !121}
!119 = !{!"dim0", i32 0}
!120 = !{!"dim1", i32 1}
!121 = !{!"dim2", i32 2}
!122 = !{!"funcArgs"}
!123 = !{!"functionType", !"KernelFunction"}
!124 = !{!"inlineDynConstants"}
!125 = !{!"inlineDynRootConstant"}
!126 = !{!"inlineDynConstantDescTable"}
!127 = !{!"m_pInterestingConstants"}
!128 = !{!"rtInfo", !129, !130, !131, !132, !133, !134, !135, !136, !137, !138, !139, !140, !141, !142, !143, !144, !148, !149, !100}
!129 = !{!"callableShaderType", !"NumberOfCallableShaderTypes"}
!130 = !{!"isContinuation", i1 false}
!131 = !{!"hasTraceRayPayload", i1 false}
!132 = !{!"hasHitAttributes", i1 false}
!133 = !{!"hasCallableData", i1 false}
!134 = !{!"ShaderStackSize", i32 0}
!135 = !{!"ShaderHash", i64 0}
!136 = !{!"ShaderName", !""}
!137 = !{!"ParentName", !""}
!138 = !{!"SlotNum", i1* null}
!139 = !{!"NOSSize", i32 0}
!140 = !{!"globalRootSignatureSize", i32 0}
!141 = !{!"Entries"}
!142 = !{!"SpillUnions"}
!143 = !{!"CustomHitAttrSizeInBytes", i32 0}
!144 = !{!"Types", !145, !146, !147}
!145 = !{!"FrameStartTys"}
!146 = !{!"ArgumentTys"}
!147 = !{!"FullFrameTys"}
!148 = !{!"Aliases"}
!149 = !{!"NumGRF", i32 0}
!150 = !{!"resAllocMD", !151, !152, !153, !154, !155}
!151 = !{!"uavsNumType", i32 0}
!152 = !{!"srvsNumType", i32 0}
!153 = !{!"samplersNumType", i32 0}
!154 = !{!"argAllocMDList"}
!155 = !{!"inlineSamplersMD"}
!156 = !{!"maxByteOffsets"}
!157 = !{!"IsInitializer", i1 false}
!158 = !{!"IsFinalizer", i1 false}
!159 = !{!"CompiledSubGroupsNumber", i32 0}
!160 = !{!"hasInlineVmeSamplers", i1 false}
!161 = !{!"localSize", i32 0}
!162 = !{!"localIDPresent", i1 false}
!163 = !{!"groupIDPresent", i1 false}
!164 = !{!"privateMemoryPerWI", i32 0}
!165 = !{!"prevFPOffset", i32 0}
!166 = !{!"globalIDPresent", i1 false}
!167 = !{!"hasSyncRTCalls", i1 false}
!168 = !{!"hasNonKernelArgLoad", i1 false}
!169 = !{!"hasNonKernelArgStore", i1 false}
!170 = !{!"hasNonKernelArgAtomic", i1 false}
!171 = !{!"UserAnnotations"}
!172 = !{!"m_OpenCLArgAddressSpaces", !173, !174, !175, !176, !177, !178, !179, !180, !181, !182}
!173 = !{!"m_OpenCLArgAddressSpacesVec[0]", i32 1}
!174 = !{!"m_OpenCLArgAddressSpacesVec[1]", i32 0}
!175 = !{!"m_OpenCLArgAddressSpacesVec[2]", i32 0}
!176 = !{!"m_OpenCLArgAddressSpacesVec[3]", i32 0}
!177 = !{!"m_OpenCLArgAddressSpacesVec[4]", i32 1}
!178 = !{!"m_OpenCLArgAddressSpacesVec[5]", i32 0}
!179 = !{!"m_OpenCLArgAddressSpacesVec[6]", i32 0}
!180 = !{!"m_OpenCLArgAddressSpacesVec[7]", i32 1}
!181 = !{!"m_OpenCLArgAddressSpacesVec[8]", i32 0}
!182 = !{!"m_OpenCLArgAddressSpacesVec[9]", i32 0}
!183 = !{!"m_OpenCLArgAccessQualifiers", !184, !185, !186, !187, !188, !189, !190, !191, !192, !193}
!184 = !{!"m_OpenCLArgAccessQualifiersVec[0]", !"none"}
!185 = !{!"m_OpenCLArgAccessQualifiersVec[1]", !"none"}
!186 = !{!"m_OpenCLArgAccessQualifiersVec[2]", !"none"}
!187 = !{!"m_OpenCLArgAccessQualifiersVec[3]", !"none"}
!188 = !{!"m_OpenCLArgAccessQualifiersVec[4]", !"none"}
!189 = !{!"m_OpenCLArgAccessQualifiersVec[5]", !"none"}
!190 = !{!"m_OpenCLArgAccessQualifiersVec[6]", !"none"}
!191 = !{!"m_OpenCLArgAccessQualifiersVec[7]", !"none"}
!192 = !{!"m_OpenCLArgAccessQualifiersVec[8]", !"none"}
!193 = !{!"m_OpenCLArgAccessQualifiersVec[9]", !"none"}
!194 = !{!"m_OpenCLArgTypes", !195, !196, !197, !198, !199, !200, !201, !202, !203, !204}
!195 = !{!"m_OpenCLArgTypesVec[0]", !"float*"}
!196 = !{!"m_OpenCLArgTypesVec[1]", !"class.sycl::_V1::range"}
!197 = !{!"m_OpenCLArgTypesVec[2]", !"class.sycl::_V1::range"}
!198 = !{!"m_OpenCLArgTypesVec[3]", !"long"}
!199 = !{!"m_OpenCLArgTypesVec[4]", !"class.sycl::_V1::ext::oneapi::bfloat16*"}
!200 = !{!"m_OpenCLArgTypesVec[5]", !"class.sycl::_V1::range"}
!201 = !{!"m_OpenCLArgTypesVec[6]", !"class.sycl::_V1::range"}
!202 = !{!"m_OpenCLArgTypesVec[7]", !"class.sycl::_V1::ext::oneapi::bfloat16*"}
!203 = !{!"m_OpenCLArgTypesVec[8]", !"class.sycl::_V1::range"}
!204 = !{!"m_OpenCLArgTypesVec[9]", !"class.sycl::_V1::range"}
!205 = !{!"m_OpenCLArgBaseTypes", !206, !207, !208, !209, !210, !211, !212, !213, !214, !215}
!206 = !{!"m_OpenCLArgBaseTypesVec[0]", !"float*"}
!207 = !{!"m_OpenCLArgBaseTypesVec[1]", !"class.sycl::_V1::range"}
!208 = !{!"m_OpenCLArgBaseTypesVec[2]", !"class.sycl::_V1::range"}
!209 = !{!"m_OpenCLArgBaseTypesVec[3]", !"long"}
!210 = !{!"m_OpenCLArgBaseTypesVec[4]", !"class.sycl::_V1::ext::oneapi::bfloat16*"}
!211 = !{!"m_OpenCLArgBaseTypesVec[5]", !"class.sycl::_V1::range"}
!212 = !{!"m_OpenCLArgBaseTypesVec[6]", !"class.sycl::_V1::range"}
!213 = !{!"m_OpenCLArgBaseTypesVec[7]", !"class.sycl::_V1::ext::oneapi::bfloat16*"}
!214 = !{!"m_OpenCLArgBaseTypesVec[8]", !"class.sycl::_V1::range"}
!215 = !{!"m_OpenCLArgBaseTypesVec[9]", !"class.sycl::_V1::range"}
!216 = !{!"m_OpenCLArgTypeQualifiers", !217, !218, !219, !220, !221, !222, !223, !224, !225, !226}
!217 = !{!"m_OpenCLArgTypeQualifiersVec[0]", !""}
!218 = !{!"m_OpenCLArgTypeQualifiersVec[1]", !""}
!219 = !{!"m_OpenCLArgTypeQualifiersVec[2]", !""}
!220 = !{!"m_OpenCLArgTypeQualifiersVec[3]", !""}
!221 = !{!"m_OpenCLArgTypeQualifiersVec[4]", !""}
!222 = !{!"m_OpenCLArgTypeQualifiersVec[5]", !""}
!223 = !{!"m_OpenCLArgTypeQualifiersVec[6]", !""}
!224 = !{!"m_OpenCLArgTypeQualifiersVec[7]", !""}
!225 = !{!"m_OpenCLArgTypeQualifiersVec[8]", !""}
!226 = !{!"m_OpenCLArgTypeQualifiersVec[9]", !""}
!227 = !{!"m_OpenCLArgNames", !228, !229, !230, !231, !232, !233, !234, !235, !236, !237}
!228 = !{!"m_OpenCLArgNamesVec[0]", !""}
!229 = !{!"m_OpenCLArgNamesVec[1]", !""}
!230 = !{!"m_OpenCLArgNamesVec[2]", !""}
!231 = !{!"m_OpenCLArgNamesVec[3]", !""}
!232 = !{!"m_OpenCLArgNamesVec[4]", !""}
!233 = !{!"m_OpenCLArgNamesVec[5]", !""}
!234 = !{!"m_OpenCLArgNamesVec[6]", !""}
!235 = !{!"m_OpenCLArgNamesVec[7]", !""}
!236 = !{!"m_OpenCLArgNamesVec[8]", !""}
!237 = !{!"m_OpenCLArgNamesVec[9]", !""}
!238 = !{!"m_OpenCLArgScalarAsPointers"}
!239 = !{!"m_OptsToDisablePerFunc"}
!240 = !{!"pushInfo", !241, !242, !243, !247, !248, !249, !250, !251, !252, !253, !254, !267, !268, !269, !270}
!241 = !{!"pushableAddresses"}
!242 = !{!"bindlessPushInfo"}
!243 = !{!"dynamicBufferInfo", !244, !245, !246}
!244 = !{!"firstIndex", i32 0}
!245 = !{!"numOffsets", i32 0}
!246 = !{!"forceDisabled", i1 false}
!247 = !{!"MaxNumberOfPushedBuffers", i32 0}
!248 = !{!"inlineConstantBufferSlot", i32 -1}
!249 = !{!"inlineConstantBufferOffset", i32 -1}
!250 = !{!"inlineConstantBufferGRFOffset", i32 -1}
!251 = !{!"constants"}
!252 = !{!"inputs"}
!253 = !{!"constantReg"}
!254 = !{!"simplePushInfoArr", !255, !264, !265, !266}
!255 = !{!"simplePushInfoArrVec[0]", !256, !257, !258, !259, !260, !261, !262, !263}
!256 = !{!"cbIdx", i32 0}
!257 = !{!"pushableAddressGrfOffset", i32 -1}
!258 = !{!"pushableOffsetGrfOffset", i32 -1}
!259 = !{!"offset", i32 0}
!260 = !{!"size", i32 0}
!261 = !{!"isStateless", i1 false}
!262 = !{!"isBindless", i1 false}
!263 = !{!"simplePushLoads"}
!264 = !{!"simplePushInfoArrVec[1]", !256, !257, !258, !259, !260, !261, !262, !263}
!265 = !{!"simplePushInfoArrVec[2]", !256, !257, !258, !259, !260, !261, !262, !263}
!266 = !{!"simplePushInfoArrVec[3]", !256, !257, !258, !259, !260, !261, !262, !263}
!267 = !{!"simplePushBufferUsed", i32 0}
!268 = !{!"pushAnalysisWIInfos"}
!269 = !{!"inlineRTGlobalPtrOffset", i32 0}
!270 = !{!"rtSyncSurfPtrOffset", i32 0}
!271 = !{!"WaEnableICBPromotion", i1 false}
!272 = !{!"vsInfo", !273, !274, !275}
!273 = !{!"DrawIndirectBufferIndex", i32 -1}
!274 = !{!"vertexReordering", i32 -1}
!275 = !{!"MaxNumOfOutputs", i32 0}
!276 = !{!"hsInfo", !277, !278}
!277 = !{!"numPatchAttributesPatchBaseName", !""}
!278 = !{!"numVertexAttributesPatchBaseName", !""}
!279 = !{!"dsInfo", !275}
!280 = !{!"gsInfo", !275}
!281 = !{!"psInfo", !282, !283, !284, !285, !286, !287, !288, !289, !290, !291, !292, !293, !294, !295, !296, !297, !298, !299, !300, !301, !302, !303, !304, !305, !306, !307, !308, !309, !310, !311, !312, !313, !314, !315}
!282 = !{!"BlendStateDisabledMask", i8 0}
!283 = !{!"SkipSrc0Alpha", i1 false}
!284 = !{!"DualSourceBlendingDisabled", i1 false}
!285 = !{!"ForceEnableSimd32", i1 false}
!286 = !{!"outputDepth", i1 false}
!287 = !{!"outputStencil", i1 false}
!288 = !{!"outputMask", i1 false}
!289 = !{!"blendToFillEnabled", i1 false}
!290 = !{!"forceEarlyZ", i1 false}
!291 = !{!"hasVersionedLoop", i1 false}
!292 = !{!"forceSingleSourceRTWAfterDualSourceRTW", i1 false}
!293 = !{!"requestCPSizeRelevant", i1 false}
!294 = !{!"requestCPSize", i1 false}
!295 = !{!"texelMaskFastClearMode", !"Disabled"}
!296 = !{!"NumSamples", i8 0}
!297 = !{!"blendOptimizationMode"}
!298 = !{!"colorOutputMask"}
!299 = !{!"ProvokingVertexModeNosIndex", i32 0}
!300 = !{!"ProvokingVertexModeNosPatch", !""}
!301 = !{!"ProvokingVertexModeLast", !"Negative"}
!302 = !{!"VertexAttributesBypass", i1 false}
!303 = !{!"LegacyBaryAssignmentDisableLinear", i1 false}
!304 = !{!"LegacyBaryAssignmentDisableLinearNoPerspective", i1 false}
!305 = !{!"LegacyBaryAssignmentDisableLinearCentroid", i1 false}
!306 = !{!"LegacyBaryAssignmentDisableLinearNoPerspectiveCentroid", i1 false}
!307 = !{!"LegacyBaryAssignmentDisableLinearSample", i1 false}
!308 = !{!"LegacyBaryAssignmentDisableLinearNoPerspectiveSample", i1 false}
!309 = !{!"MeshShaderWAPerPrimitiveUserDataEnable", !"Negative"}
!310 = !{!"meshShaderWAPerPrimitiveUserDataEnablePatchName", !""}
!311 = !{!"generatePatchesForRTWriteSends", i1 false}
!312 = !{!"forceVMask", i1 false}
!313 = !{!"WaDisableVRS", i1 false}
!314 = !{!"RelaxMemoryVisibilityFromPSOrdering", i1 false}
!315 = !{!"WaEnableVMaskUnderNonUnifromCF", i1 false}
!316 = !{!"csInfo", !317, !318, !319, !320, !321, !29, !30, !322, !323, !324, !325, !326, !327, !328, !329, !330, !331, !332, !333, !334, !62, !335, !336, !337, !338, !339, !340, !341}
!317 = !{!"maxWorkGroupSize", i32 0}
!318 = !{!"waveSize", i32 0}
!319 = !{!"ComputeShaderSecondCompile"}
!320 = !{!"forcedSIMDSize", i8 0}
!321 = !{!"forceTotalGRFNum", i32 0}
!322 = !{!"forceSpillCompression", i1 false}
!323 = !{!"allowLowerSimd", i1 false}
!324 = !{!"disableSimd32Slicing", i1 false}
!325 = !{!"disableSplitOnSpill", i1 false}
!326 = !{!"enableNewSpillCostFunction", i1 false}
!327 = !{!"forceVISAPreSched", i1 false}
!328 = !{!"forceUniformBuffer", i1 false}
!329 = !{!"forceUniformSurfaceSampler", i1 false}
!330 = !{!"disableLocalIdOrderOptimizations", i1 false}
!331 = !{!"disableDispatchAlongY", i1 false}
!332 = !{!"neededThreadIdLayout", i1* null}
!333 = !{!"forceTileYWalk", i1 false}
!334 = !{!"atomicBranch", i32 0}
!335 = !{!"disableEarlyOut", i1 false}
!336 = !{!"walkOrderEnabled", i1 false}
!337 = !{!"walkOrderOverride", i32 0}
!338 = !{!"ResForHfPacking"}
!339 = !{!"hasWaveMatrix", i1 false}
!340 = !{!"constantFoldSimdSize", i1 false}
!341 = !{!"isNodeShader", i1 false}
!342 = !{!"msInfo", !343, !344, !345, !346, !347, !348, !349, !350, !351, !352, !353, !301, !299, !354}
!343 = !{!"PrimitiveTopology", i32 3}
!344 = !{!"MaxNumOfPrimitives", i32 0}
!345 = !{!"MaxNumOfVertices", i32 0}
!346 = !{!"MaxNumOfPerPrimitiveOutputs", i32 0}
!347 = !{!"MaxNumOfPerVertexOutputs", i32 0}
!348 = !{!"WorkGroupSize", i32 0}
!349 = !{!"WorkGroupMemorySizeInBytes", i32 0}
!350 = !{!"IndexFormat", i32 6}
!351 = !{!"SubgroupSize", i32 0}
!352 = !{!"VPandRTAIndexAutostripEnable", i1 false}
!353 = !{!"MeshShaderWAPerPrimitiveUserDataEnable", i1 false}
!354 = !{!"numPrimitiveAttributesPatchBaseName", !""}
!355 = !{!"taskInfo", !275, !348, !349, !351}
!356 = !{!"NBarrierCnt", i32 0}
!357 = !{!"rtInfo", !358, !359, !360, !361, !362, !363, !364, !365, !366, !367, !368, !369, !370, !371}
!358 = !{!"RayQueryAllocSizeInBytes", i32 0}
!359 = !{!"NumContinuations", i32 0}
!360 = !{!"RTAsyncStackAddrspace", i32 -1}
!361 = !{!"RTAsyncStackSurfaceStateOffset", i1* null}
!362 = !{!"SWHotZoneAddrspace", i32 -1}
!363 = !{!"SWHotZoneSurfaceStateOffset", i1* null}
!364 = !{!"SWStackAddrspace", i32 -1}
!365 = !{!"SWStackSurfaceStateOffset", i1* null}
!366 = !{!"RTSyncStackAddrspace", i32 -1}
!367 = !{!"RTSyncStackSurfaceStateOffset", i1* null}
!368 = !{!"doSyncDispatchRays", i1 false}
!369 = !{!"MemStyle", !"Xe"}
!370 = !{!"GlobalDataStyle", !"Xe"}
!371 = !{!"NeedsBTD", i1 true}
!372 = !{!"EnableTextureIndirection", i1 false}
!373 = !{!"EnableSamplerIndirection", i1 false}
!374 = !{!"samplerStateStride", i32 0}
!375 = !{!"samplerStateOffset", i32 0}
!376 = !{!"textureStateStride", i32 0}
!377 = !{!"textureStateOffset", i32 0}
!378 = !{!"CurUniqueIndirectIdx", i32 0}
!379 = !{!"inlineDynTextures"}
!380 = !{!"inlineResInfoData"}
!381 = !{!"immConstant", !382, !383, !384}
!382 = !{!"data"}
!383 = !{!"sizes"}
!384 = !{!"zeroIdxs"}
!385 = !{!"stringConstants"}
!386 = !{!"inlineBuffers", !387, !391, !392}
!387 = !{!"inlineBuffersVec[0]", !388, !389, !390}
!388 = !{!"alignment", i32 0}
!389 = !{!"allocSize", i64 0}
!390 = !{!"Buffer"}
!391 = !{!"inlineBuffersVec[1]", !388, !389, !390}
!392 = !{!"inlineBuffersVec[2]", !388, !389, !390}
!393 = !{!"GlobalPointerProgramBinaryInfos"}
!394 = !{!"ConstantPointerProgramBinaryInfos"}
!395 = !{!"GlobalBufferAddressRelocInfo"}
!396 = !{!"ConstantBufferAddressRelocInfo"}
!397 = !{!"forceLscCacheList"}
!398 = !{!"SrvMap"}
!399 = !{!"RootConstantBufferOffsetInBytes"}
!400 = !{!"RasterizerOrderedByteAddressBuffer"}
!401 = !{!"RasterizerOrderedViews"}
!402 = !{!"MinNOSPushConstantSize", i32 0}
!403 = !{!"inlineProgramScopeOffsets"}
!404 = !{!"shaderData", !405}
!405 = !{!"numReplicas", i32 0}
!406 = !{!"URBInfo", !407, !408, !409}
!407 = !{!"has64BVertexHeaderInput", i1 false}
!408 = !{!"has64BVertexHeaderOutput", i1 false}
!409 = !{!"hasVertexHeader", i1 true}
!410 = !{!"m_ForcePullModel", i1 false}
!411 = !{!"UseBindlessImage", i1 false}
!412 = !{!"enableRangeReduce", i1 false}
!413 = !{!"disableNewTrigFuncRangeReduction", i1 false}
!414 = !{!"enableFRemToSRemOpt", i1 false}
!415 = !{!"enableSampleptrToLdmsptrSample0", i1 false}
!416 = !{!"enableSampleLptrToLdmsptrSample0", i1 false}
!417 = !{!"WaForceSIMD32MicropolyRasterize", i1 false}
!418 = !{!"allowMatchMadOptimizationforVS", i1 false}
!419 = !{!"disableMatchMadOptimizationForCS", i1 false}
!420 = !{!"disableMemOptforNegativeOffsetLoads", i1 false}
!421 = !{!"enableThreeWayLoadSpiltOpt", i1 false}
!422 = !{!"statefulResourcesNotAliased", i1 false}
!423 = !{!"disableMixMode", i1 false}
!424 = !{!"genericAccessesResolved", i1 false}
!425 = !{!"disableSeparateSpillPvtScratchSpace", i1 false}
!426 = !{!"disableSeparateScratchWA", i1 false}
!427 = !{!"PrivateMemoryPerFG"}
!428 = !{!"m_OptsToDisable"}
!429 = !{!"capabilities", !430}
!430 = !{!"globalVariableDecorationsINTEL", i1 false}
!431 = !{!"m_ShaderResourceViewMcsMask", !432, !433}
!432 = !{!"m_ShaderResourceViewMcsMaskVec[0]", i64 0}
!433 = !{!"m_ShaderResourceViewMcsMaskVec[1]", i64 0}
!434 = !{!"computedDepthMode", i32 0}
!435 = !{!"isHDCFastClearShader", i1 false}
!436 = !{!"argRegisterReservations", !437}
!437 = !{!"argRegisterReservationsVec[0]", i32 0}
!438 = !{!"SIMD16_SpillThreshold", i8 0}
!439 = !{!"SIMD32_SpillThreshold", i8 0}
!440 = !{!"m_CacheControlOption", !441, !442, !443, !444}
!441 = !{!"LscLoadCacheControlOverride", i8 0}
!442 = !{!"LscStoreCacheControlOverride", i8 0}
!443 = !{!"TgmLoadCacheControlOverride", i8 0}
!444 = !{!"TgmStoreCacheControlOverride", i8 0}
!445 = !{i32 2, i32 0}
!446 = !{!"clang version 14.0.5"}
!447 = !{i32 1, !"wchar_size", i32 4}
!448 = !{i32 1, i32 0, i32 0, i32 0, i32 1, i32 0, i32 0, i32 1, i32 0, i32 0}
!449 = !{!"none", !"none", !"none", !"none", !"none", !"none", !"none", !"none", !"none", !"none"}
!450 = !{!"float*", !"class.sycl::_V1::range", !"class.sycl::_V1::range", !"long", !"class.sycl::_V1::ext::oneapi::bfloat16*", !"class.sycl::_V1::range", !"class.sycl::_V1::range", !"class.sycl::_V1::ext::oneapi::bfloat16*", !"class.sycl::_V1::range", !"class.sycl::_V1::range"}
!451 = !{!"", !"", !"", !"", !"", !"", !"", !"", !"", !""}
!452 = !{!453, !455, !455, !458, !459, !455, !455, !459, !455, !455}
!453 = !{!454}
!454 = !{i32 44, i32 4}
!455 = !{!456, !457}
!456 = !{i32 38, i32 2}
!457 = !{i32 44, i32 8}
!458 = !{}
!459 = !{!460}
!460 = !{i32 44, i32 2}
!461 = !{!462}
!462 = !{i32 4469}
!463 = !{!464, !464, i64 0}
!464 = !{!"int", !465, i64 0}
!465 = !{!"omnipotent char", !466, i64 0}
!466 = !{!"Simple C/C++ TBAA"}
!467 = !{!462, !468}
!468 = !{i32 4470}
