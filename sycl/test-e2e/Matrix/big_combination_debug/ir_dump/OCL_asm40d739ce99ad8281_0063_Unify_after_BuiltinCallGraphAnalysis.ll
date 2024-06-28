; ------------------------------------------------
; OCL_asm40d739ce99ad8281_0063_Unify_after_BuiltinCallGraphAnalysis.ll
; LLVM major version: 14
; ------------------------------------------------
target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v16:16:16-v24:32:32-v32:32:32-v48:64:64-v64:64:64-v96:128:128-v128:128:128-v192:256:256-v256:256:256-v512:512:512-v1024:1024:1024-n8:16:32"
target triple = "spir64-unknown-unknown"

%"class.sycl::_V1::ext::oneapi::bfloat16" = type { i16 }
%"class.sycl::_V1::range" = type { %"class.sycl::_V1::detail::array" }
%"class.sycl::_V1::detail::array" = type { [2 x i64] }

; Function Attrs: convergent nounwind
define spir_kernel void @_ZTS7imatrixIfLm32ELm32ELm16EE(float addrspace(1)* align 4 %0, %"class.sycl::_V1::range"* byval(%"class.sycl::_V1::range") align 8 %1, %"class.sycl::_V1::range"* byval(%"class.sycl::_V1::range") align 8 %2, i64 %3, %"class.sycl::_V1::ext::oneapi::bfloat16" addrspace(1)* align 2 %4, %"class.sycl::_V1::range"* byval(%"class.sycl::_V1::range") align 8 %5, %"class.sycl::_V1::range"* byval(%"class.sycl::_V1::range") align 8 %6, %"class.sycl::_V1::ext::oneapi::bfloat16" addrspace(1)* align 2 %7, %"class.sycl::_V1::range"* byval(%"class.sycl::_V1::range") align 8 %8, %"class.sycl::_V1::range"* byval(%"class.sycl::_V1::range") align 8 %9) #0 !kernel_arg_addr_space !469 !kernel_arg_access_qual !470 !kernel_arg_type !471 !kernel_arg_type_qual !472 !kernel_arg_base_type !471 !kernel_arg_name !472 !spirv.ParameterDecorations !473 {
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
  %72 = sub nsw i64 %56, %70, !spirv.Decorations !482
  %73 = call spir_func i32 @__builtin_IB_get_local_id_x() #4
  %74 = icmp ult i32 %73, 65536
  call void @llvm.assume(i1 %74) #5
  %75 = zext i32 %73 to i64
  %76 = icmp ult i64 %75, 2147483648
  call void @llvm.assume(i1 %76)
  %77 = sub nsw i64 %66, %75, !spirv.Decorations !482
  %78 = add i64 %23, %22
  %79 = sub i64 0, %78
  %80 = getelementptr inbounds float, float addrspace(1)* %25, i64 %79
  %81 = shl nsw i64 %72, 11, !spirv.Decorations !482
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
  %142 = load i32, i32 addrspace(1)* %141, align 4, !tbaa !484
  br label %143

143:                                              ; preds = %137, %132
  %144 = phi i32 [ %142, %137 ], [ 0, %132 ]
  %145 = zext i32 %133 to i64
  %146 = getelementptr inbounds i32, i32* %126, i64 %145
  store i32 %144, i32* %146, align 4, !tbaa !484
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
  %158 = shl nsw i64 %72, 10, !spirv.Decorations !482
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
  %169 = shl nuw nsw i32 %166, 4, !spirv.Decorations !488
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
  store i32 %215, i32* %216, align 4, !tbaa !484
  %217 = add nuw nsw i32 %211, 1
  %218 = icmp ult i32 %211, 31
  br i1 %218, label %210, label %__builtin_spriv_OpJointMatrixLoadINTEL_PackedA_RowMajor_32x16_i16_32_global_v8i8_pi32_i32.24.exit

__builtin_spriv_OpJointMatrixLoadINTEL_PackedA_RowMajor_32x16_i16_32_global_v8i8_pi32_i32.24.exit: ; preds = %210
  %219 = shl nuw nsw i64 %170, 6, !spirv.Decorations !488
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
  %277 = load i32, i32 addrspace(1)* %276, align 4, !tbaa !484
  br label %278

278:                                              ; preds = %272, %267
  %279 = phi i32 [ %277, %272 ], [ 0, %267 ]
  %280 = zext i32 %268 to i64
  %281 = getelementptr inbounds i32, i32* %261, i64 %280
  store i32 %279, i32* %281, align 4, !tbaa !484
  %282 = add nuw nsw i32 %268, 1
  %283 = icmp ult i32 %268, 31
  br i1 %283, label %267, label %__builtin_spriv_OpJointMatrixLoadINTEL_PackedB_PackedB_16x32_i16_32_global_v8i8_pi32_i32.25.exit

__builtin_spriv_OpJointMatrixLoadINTEL_PackedB_PackedB_16x32_i16_32_global_v8i8_pi32_i32.25.exit: ; preds = %278
  %284 = add nuw nsw i32 %166, 1, !spirv.Decorations !488
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
  %340 = load i32, i32* %339, align 4, !tbaa !484
  %341 = sext i32 %335 to i64
  %342 = mul nsw i64 %341, 64
  %343 = add nsw i64 %342, %331
  %344 = getelementptr inbounds i32, i32 addrspace(1)* %323, i64 %343
  store i32 %340, i32 addrspace(1)* %344, align 4, !tbaa !484
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
!IGCMetadata = !{!36}
!opencl.ocl.version = !{!466, !466, !466, !466, !466}
!opencl.spir.version = !{!466, !466, !466, !466, !466}
!llvm.ident = !{!467, !467, !467, !467, !467}
!llvm.module.flags = !{!468}

!0 = !{i32 2, i32 2}
!1 = !{i32 4, i32 100000}
!2 = !{i16 6, i16 14}
!3 = !{void (float addrspace(1)*, %"class.sycl::_V1::range"*, %"class.sycl::_V1::range"*, i64, %"class.sycl::_V1::ext::oneapi::bfloat16" addrspace(1)*, %"class.sycl::_V1::range"*, %"class.sycl::_V1::range"*, %"class.sycl::_V1::ext::oneapi::bfloat16" addrspace(1)*, %"class.sycl::_V1::range"*, %"class.sycl::_V1::range"*)* @_ZTS7imatrixIfLm32ELm32ELm16EE, !4}
!4 = !{!5, !6, !35}
!5 = !{!"function_type", i32 0}
!6 = !{!"implicit_arg_desc", !7, !8, !9, !10, !11, !12, !13, !14, !15, !18, !20, !22, !23, !25, !26, !28, !29, !31, !32, !34}
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
!35 = !{!"sub_group_size", i32 8}
!36 = !{!"ModuleMD", !37, !38, !135, !261, !292, !293, !297, !300, !301, !302, !337, !363, !376, !377, !378, !393, !394, !395, !396, !397, !398, !399, !400, !401, !402, !406, !407, !414, !415, !416, !417, !418, !419, !420, !421, !422, !423, !424, !425, !427, !431, !432, !433, !434, !435, !436, !437, !438, !439, !440, !441, !442, !443, !444, !445, !446, !447, !185, !448, !449, !450, !452, !455, !456, !457, !459, !460, !461}
!37 = !{!"isPrecise", i1 false}
!38 = !{!"compOpt", !39, !40, !41, !42, !43, !44, !45, !46, !47, !48, !49, !50, !51, !52, !53, !54, !55, !56, !57, !58, !59, !60, !61, !62, !63, !64, !65, !66, !67, !68, !69, !70, !71, !72, !73, !74, !75, !76, !77, !78, !79, !80, !81, !82, !83, !84, !85, !86, !87, !88, !89, !90, !91, !92, !93, !94, !95, !96, !97, !98, !99, !100, !101, !102, !103, !104, !105, !106, !107, !108, !109, !110, !111, !112, !113, !114, !115, !116, !117, !118, !119, !120, !121, !122, !123, !124, !125, !126, !127, !128, !129, !130, !131, !132, !133, !134}
!39 = !{!"DenormsAreZero", i1 false}
!40 = !{!"BFTFDenormsAreZero", i1 false}
!41 = !{!"CorrectlyRoundedDivSqrt", i1 false}
!42 = !{!"OptDisable", i1 false}
!43 = !{!"MadEnable", i1 true}
!44 = !{!"NoSignedZeros", i1 false}
!45 = !{!"NoNaNs", i1 false}
!46 = !{!"FloatRoundingMode", i32 0}
!47 = !{!"FloatCvtIntRoundingMode", i32 3}
!48 = !{!"LoadCacheDefault", i32 4}
!49 = !{!"StoreCacheDefault", i32 7}
!50 = !{!"VISAPreSchedRPThreshold", i32 0}
!51 = !{!"SetLoopUnrollThreshold", i32 0}
!52 = !{!"UnsafeMathOptimizations", i1 false}
!53 = !{!"disableCustomUnsafeOpts", i1 false}
!54 = !{!"disableReducePow", i1 false}
!55 = !{!"disableSqrtOpt", i1 false}
!56 = !{!"FiniteMathOnly", i1 false}
!57 = !{!"FastRelaxedMath", i1 false}
!58 = !{!"DashGSpecified", i1 false}
!59 = !{!"FastCompilation", i1 false}
!60 = !{!"UseScratchSpacePrivateMemory", i1 true}
!61 = !{!"RelaxedBuiltins", i1 false}
!62 = !{!"SubgroupIndependentForwardProgressRequired", i1 true}
!63 = !{!"GreaterThan2GBBufferRequired", i1 true}
!64 = !{!"GreaterThan4GBBufferRequired", i1 false}
!65 = !{!"DisableA64WA", i1 false}
!66 = !{!"ForceEnableA64WA", i1 false}
!67 = !{!"PushConstantsEnable", i1 true}
!68 = !{!"HasPositivePointerOffset", i1 false}
!69 = !{!"HasBufferOffsetArg", i1 true}
!70 = !{!"BufferOffsetArgOptional", i1 true}
!71 = !{!"replaceGlobalOffsetsByZero", i1 false}
!72 = !{!"forcePixelShaderSIMDMode", i32 0}
!73 = !{!"ForceGeomFFShaderSIMDMode", i32 0}
!74 = !{!"pixelShaderDoNotAbortOnSpill", i1 false}
!75 = !{!"UniformWGS", i1 false}
!76 = !{!"disableVertexComponentPacking", i1 false}
!77 = !{!"disablePartialVertexComponentPacking", i1 false}
!78 = !{!"PreferBindlessImages", i1 false}
!79 = !{!"UseBindlessMode", i1 false}
!80 = !{!"UseLegacyBindlessMode", i1 true}
!81 = !{!"disableMathRefactoring", i1 false}
!82 = !{!"atomicBranch", i1 false}
!83 = !{!"spillCompression", i1 false}
!84 = !{!"DisableEarlyOut", i1 false}
!85 = !{!"ForceInt32DivRemEmu", i1 false}
!86 = !{!"ForceInt32DivRemEmuSP", i1 false}
!87 = !{!"WaveIntrinsicUsed", i1 false}
!88 = !{!"DisableMultiPolyPS", i1 false}
!89 = !{!"NeedTexture3DLODWA", i1 false}
!90 = !{!"DisableFastestSingleCSSIMD", i1 false}
!91 = !{!"DisableFastestLinearScan", i1 false}
!92 = !{!"UseStatelessforPrivateMemory", i1 false}
!93 = !{!"EnableTakeGlobalAddress", i1 false}
!94 = !{!"IsLibraryCompilation", i1 false}
!95 = !{!"LibraryCompileSIMDSize", i32 0}
!96 = !{!"FastVISACompile", i1 false}
!97 = !{!"MatchSinCosPi", i1 false}
!98 = !{!"ExcludeIRFromZEBinary", i1 false}
!99 = !{!"EmitZeBinVISASections", i1 false}
!100 = !{!"FP64GenEmulationEnabled", i1 false}
!101 = !{!"FP64GenConvEmulationEnabled", i1 false}
!102 = !{!"allowDisableRematforCS", i1 false}
!103 = !{!"DisableIncSpillCostAllAddrTaken", i1 false}
!104 = !{!"DisableCPSOmaskWA", i1 false}
!105 = !{!"DisableFastestGopt", i1 false}
!106 = !{!"WaForceHalfPromotionComputeShader", i1 false}
!107 = !{!"WaForceHalfPromotionPixelVertexShader", i1 false}
!108 = !{!"DisableConstantCoalescing", i1 false}
!109 = !{!"EnableUndefAlphaOutputAsRed", i1 true}
!110 = !{!"WaEnableALTModeVisaWA", i1 false}
!111 = !{!"WaEnableAtomicWaveFusion", i1 false}
!112 = !{!"WaEnableAtomicWaveFusionNonNullResource", i1 false}
!113 = !{!"WaEnableAtomicWaveFusionStateless", i1 false}
!114 = !{!"WaEnableAtomicWaveFusionTyped", i1 false}
!115 = !{!"ForceCBThroughSampler3D", i1 false}
!116 = !{!"WaStoreRawVectorToTypedWrite", i1 false}
!117 = !{!"WaLoadRawVectorToTypedRead", i1 false}
!118 = !{!"WaZeroSLMBeforeUse", i1 false}
!119 = !{!"WaFlagGroupTypedUAVGloballyCoherent", i1 false}
!120 = !{!"NewSpillCostFunction", i1 false}
!121 = !{!"EnableVRT", i1 false}
!122 = !{!"ForceLargeGRFNum4RQ", i1 false}
!123 = !{!"Enable2xGRFRetry", i1 false}
!124 = !{!"Detect2xGRFCandidate", i1 false}
!125 = !{!"EnableURBWritesMerging", i1 true}
!126 = !{!"DisableEUFusion", i1 false}
!127 = !{!"DisableFDivToFMulInvOpt", i1 false}
!128 = !{!"initializePhiSampleSourceWA", i1 false}
!129 = !{!"WaDisableSubspanUseNoMaskForCB", i1 false}
!130 = !{!"DisableLoosenSimd32Occu", i1 false}
!131 = !{!"FastestS1Options", i32 0}
!132 = !{!"DisableFastestForWaveIntrinsicsCS", i1 false}
!133 = !{!"ForceLinearWalkOnLinearUAV", i1 false}
!134 = !{!"DisableLscSamplerRouting", i1 false}
!135 = !{!"FuncMD", !136, !137}
!136 = !{!"FuncMDMap[0]", void (float addrspace(1)*, %"class.sycl::_V1::range"*, %"class.sycl::_V1::range"*, i64, %"class.sycl::_V1::ext::oneapi::bfloat16" addrspace(1)*, %"class.sycl::_V1::range"*, %"class.sycl::_V1::range"*, %"class.sycl::_V1::ext::oneapi::bfloat16" addrspace(1)*, %"class.sycl::_V1::range"*, %"class.sycl::_V1::range"*)* @_ZTS7imatrixIfLm32ELm32ELm16EE}
!137 = !{!"FuncMDValue[0]", !138, !139, !143, !144, !145, !146, !147, !148, !149, !171, !177, !178, !179, !180, !181, !182, !183, !184, !185, !186, !187, !188, !189, !190, !191, !192, !193, !204, !215, !226, !237, !248, !259, !260}
!138 = !{!"localOffsets"}
!139 = !{!"workGroupWalkOrder", !140, !141, !142}
!140 = !{!"dim0", i32 0}
!141 = !{!"dim1", i32 1}
!142 = !{!"dim2", i32 2}
!143 = !{!"funcArgs"}
!144 = !{!"functionType", !"KernelFunction"}
!145 = !{!"inlineDynConstants"}
!146 = !{!"inlineDynRootConstant"}
!147 = !{!"inlineDynConstantDescTable"}
!148 = !{!"m_pInterestingConstants"}
!149 = !{!"rtInfo", !150, !151, !152, !153, !154, !155, !156, !157, !158, !159, !160, !161, !162, !163, !164, !165, !169, !170, !121}
!150 = !{!"callableShaderType", !"NumberOfCallableShaderTypes"}
!151 = !{!"isContinuation", i1 false}
!152 = !{!"hasTraceRayPayload", i1 false}
!153 = !{!"hasHitAttributes", i1 false}
!154 = !{!"hasCallableData", i1 false}
!155 = !{!"ShaderStackSize", i32 0}
!156 = !{!"ShaderHash", i64 0}
!157 = !{!"ShaderName", !""}
!158 = !{!"ParentName", !""}
!159 = !{!"SlotNum", i1* null}
!160 = !{!"NOSSize", i32 0}
!161 = !{!"globalRootSignatureSize", i32 0}
!162 = !{!"Entries"}
!163 = !{!"SpillUnions"}
!164 = !{!"CustomHitAttrSizeInBytes", i32 0}
!165 = !{!"Types", !166, !167, !168}
!166 = !{!"FrameStartTys"}
!167 = !{!"ArgumentTys"}
!168 = !{!"FullFrameTys"}
!169 = !{!"Aliases"}
!170 = !{!"NumGRF", i32 0}
!171 = !{!"resAllocMD", !172, !173, !174, !175, !176}
!172 = !{!"uavsNumType", i32 0}
!173 = !{!"srvsNumType", i32 0}
!174 = !{!"samplersNumType", i32 0}
!175 = !{!"argAllocMDList"}
!176 = !{!"inlineSamplersMD"}
!177 = !{!"maxByteOffsets"}
!178 = !{!"IsInitializer", i1 false}
!179 = !{!"IsFinalizer", i1 false}
!180 = !{!"CompiledSubGroupsNumber", i32 0}
!181 = !{!"hasInlineVmeSamplers", i1 false}
!182 = !{!"localSize", i32 0}
!183 = !{!"localIDPresent", i1 false}
!184 = !{!"groupIDPresent", i1 false}
!185 = !{!"privateMemoryPerWI", i32 0}
!186 = !{!"prevFPOffset", i32 0}
!187 = !{!"globalIDPresent", i1 false}
!188 = !{!"hasSyncRTCalls", i1 false}
!189 = !{!"hasNonKernelArgLoad", i1 false}
!190 = !{!"hasNonKernelArgStore", i1 false}
!191 = !{!"hasNonKernelArgAtomic", i1 false}
!192 = !{!"UserAnnotations"}
!193 = !{!"m_OpenCLArgAddressSpaces", !194, !195, !196, !197, !198, !199, !200, !201, !202, !203}
!194 = !{!"m_OpenCLArgAddressSpacesVec[0]", i32 1}
!195 = !{!"m_OpenCLArgAddressSpacesVec[1]", i32 0}
!196 = !{!"m_OpenCLArgAddressSpacesVec[2]", i32 0}
!197 = !{!"m_OpenCLArgAddressSpacesVec[3]", i32 0}
!198 = !{!"m_OpenCLArgAddressSpacesVec[4]", i32 1}
!199 = !{!"m_OpenCLArgAddressSpacesVec[5]", i32 0}
!200 = !{!"m_OpenCLArgAddressSpacesVec[6]", i32 0}
!201 = !{!"m_OpenCLArgAddressSpacesVec[7]", i32 1}
!202 = !{!"m_OpenCLArgAddressSpacesVec[8]", i32 0}
!203 = !{!"m_OpenCLArgAddressSpacesVec[9]", i32 0}
!204 = !{!"m_OpenCLArgAccessQualifiers", !205, !206, !207, !208, !209, !210, !211, !212, !213, !214}
!205 = !{!"m_OpenCLArgAccessQualifiersVec[0]", !"none"}
!206 = !{!"m_OpenCLArgAccessQualifiersVec[1]", !"none"}
!207 = !{!"m_OpenCLArgAccessQualifiersVec[2]", !"none"}
!208 = !{!"m_OpenCLArgAccessQualifiersVec[3]", !"none"}
!209 = !{!"m_OpenCLArgAccessQualifiersVec[4]", !"none"}
!210 = !{!"m_OpenCLArgAccessQualifiersVec[5]", !"none"}
!211 = !{!"m_OpenCLArgAccessQualifiersVec[6]", !"none"}
!212 = !{!"m_OpenCLArgAccessQualifiersVec[7]", !"none"}
!213 = !{!"m_OpenCLArgAccessQualifiersVec[8]", !"none"}
!214 = !{!"m_OpenCLArgAccessQualifiersVec[9]", !"none"}
!215 = !{!"m_OpenCLArgTypes", !216, !217, !218, !219, !220, !221, !222, !223, !224, !225}
!216 = !{!"m_OpenCLArgTypesVec[0]", !"float*"}
!217 = !{!"m_OpenCLArgTypesVec[1]", !"class.sycl::_V1::range"}
!218 = !{!"m_OpenCLArgTypesVec[2]", !"class.sycl::_V1::range"}
!219 = !{!"m_OpenCLArgTypesVec[3]", !"long"}
!220 = !{!"m_OpenCLArgTypesVec[4]", !"class.sycl::_V1::ext::oneapi::bfloat16*"}
!221 = !{!"m_OpenCLArgTypesVec[5]", !"class.sycl::_V1::range"}
!222 = !{!"m_OpenCLArgTypesVec[6]", !"class.sycl::_V1::range"}
!223 = !{!"m_OpenCLArgTypesVec[7]", !"class.sycl::_V1::ext::oneapi::bfloat16*"}
!224 = !{!"m_OpenCLArgTypesVec[8]", !"class.sycl::_V1::range"}
!225 = !{!"m_OpenCLArgTypesVec[9]", !"class.sycl::_V1::range"}
!226 = !{!"m_OpenCLArgBaseTypes", !227, !228, !229, !230, !231, !232, !233, !234, !235, !236}
!227 = !{!"m_OpenCLArgBaseTypesVec[0]", !"float*"}
!228 = !{!"m_OpenCLArgBaseTypesVec[1]", !"class.sycl::_V1::range"}
!229 = !{!"m_OpenCLArgBaseTypesVec[2]", !"class.sycl::_V1::range"}
!230 = !{!"m_OpenCLArgBaseTypesVec[3]", !"long"}
!231 = !{!"m_OpenCLArgBaseTypesVec[4]", !"class.sycl::_V1::ext::oneapi::bfloat16*"}
!232 = !{!"m_OpenCLArgBaseTypesVec[5]", !"class.sycl::_V1::range"}
!233 = !{!"m_OpenCLArgBaseTypesVec[6]", !"class.sycl::_V1::range"}
!234 = !{!"m_OpenCLArgBaseTypesVec[7]", !"class.sycl::_V1::ext::oneapi::bfloat16*"}
!235 = !{!"m_OpenCLArgBaseTypesVec[8]", !"class.sycl::_V1::range"}
!236 = !{!"m_OpenCLArgBaseTypesVec[9]", !"class.sycl::_V1::range"}
!237 = !{!"m_OpenCLArgTypeQualifiers", !238, !239, !240, !241, !242, !243, !244, !245, !246, !247}
!238 = !{!"m_OpenCLArgTypeQualifiersVec[0]", !""}
!239 = !{!"m_OpenCLArgTypeQualifiersVec[1]", !""}
!240 = !{!"m_OpenCLArgTypeQualifiersVec[2]", !""}
!241 = !{!"m_OpenCLArgTypeQualifiersVec[3]", !""}
!242 = !{!"m_OpenCLArgTypeQualifiersVec[4]", !""}
!243 = !{!"m_OpenCLArgTypeQualifiersVec[5]", !""}
!244 = !{!"m_OpenCLArgTypeQualifiersVec[6]", !""}
!245 = !{!"m_OpenCLArgTypeQualifiersVec[7]", !""}
!246 = !{!"m_OpenCLArgTypeQualifiersVec[8]", !""}
!247 = !{!"m_OpenCLArgTypeQualifiersVec[9]", !""}
!248 = !{!"m_OpenCLArgNames", !249, !250, !251, !252, !253, !254, !255, !256, !257, !258}
!249 = !{!"m_OpenCLArgNamesVec[0]", !""}
!250 = !{!"m_OpenCLArgNamesVec[1]", !""}
!251 = !{!"m_OpenCLArgNamesVec[2]", !""}
!252 = !{!"m_OpenCLArgNamesVec[3]", !""}
!253 = !{!"m_OpenCLArgNamesVec[4]", !""}
!254 = !{!"m_OpenCLArgNamesVec[5]", !""}
!255 = !{!"m_OpenCLArgNamesVec[6]", !""}
!256 = !{!"m_OpenCLArgNamesVec[7]", !""}
!257 = !{!"m_OpenCLArgNamesVec[8]", !""}
!258 = !{!"m_OpenCLArgNamesVec[9]", !""}
!259 = !{!"m_OpenCLArgScalarAsPointers"}
!260 = !{!"m_OptsToDisablePerFunc"}
!261 = !{!"pushInfo", !262, !263, !264, !268, !269, !270, !271, !272, !273, !274, !275, !288, !289, !290, !291}
!262 = !{!"pushableAddresses"}
!263 = !{!"bindlessPushInfo"}
!264 = !{!"dynamicBufferInfo", !265, !266, !267}
!265 = !{!"firstIndex", i32 0}
!266 = !{!"numOffsets", i32 0}
!267 = !{!"forceDisabled", i1 false}
!268 = !{!"MaxNumberOfPushedBuffers", i32 0}
!269 = !{!"inlineConstantBufferSlot", i32 -1}
!270 = !{!"inlineConstantBufferOffset", i32 -1}
!271 = !{!"inlineConstantBufferGRFOffset", i32 -1}
!272 = !{!"constants"}
!273 = !{!"inputs"}
!274 = !{!"constantReg"}
!275 = !{!"simplePushInfoArr", !276, !285, !286, !287}
!276 = !{!"simplePushInfoArrVec[0]", !277, !278, !279, !280, !281, !282, !283, !284}
!277 = !{!"cbIdx", i32 0}
!278 = !{!"pushableAddressGrfOffset", i32 -1}
!279 = !{!"pushableOffsetGrfOffset", i32 -1}
!280 = !{!"offset", i32 0}
!281 = !{!"size", i32 0}
!282 = !{!"isStateless", i1 false}
!283 = !{!"isBindless", i1 false}
!284 = !{!"simplePushLoads"}
!285 = !{!"simplePushInfoArrVec[1]", !277, !278, !279, !280, !281, !282, !283, !284}
!286 = !{!"simplePushInfoArrVec[2]", !277, !278, !279, !280, !281, !282, !283, !284}
!287 = !{!"simplePushInfoArrVec[3]", !277, !278, !279, !280, !281, !282, !283, !284}
!288 = !{!"simplePushBufferUsed", i32 0}
!289 = !{!"pushAnalysisWIInfos"}
!290 = !{!"inlineRTGlobalPtrOffset", i32 0}
!291 = !{!"rtSyncSurfPtrOffset", i32 0}
!292 = !{!"WaEnableICBPromotion", i1 false}
!293 = !{!"vsInfo", !294, !295, !296}
!294 = !{!"DrawIndirectBufferIndex", i32 -1}
!295 = !{!"vertexReordering", i32 -1}
!296 = !{!"MaxNumOfOutputs", i32 0}
!297 = !{!"hsInfo", !298, !299}
!298 = !{!"numPatchAttributesPatchBaseName", !""}
!299 = !{!"numVertexAttributesPatchBaseName", !""}
!300 = !{!"dsInfo", !296}
!301 = !{!"gsInfo", !296}
!302 = !{!"psInfo", !303, !304, !305, !306, !307, !308, !309, !310, !311, !312, !313, !314, !315, !316, !317, !318, !319, !320, !321, !322, !323, !324, !325, !326, !327, !328, !329, !330, !331, !332, !333, !334, !335, !336}
!303 = !{!"BlendStateDisabledMask", i8 0}
!304 = !{!"SkipSrc0Alpha", i1 false}
!305 = !{!"DualSourceBlendingDisabled", i1 false}
!306 = !{!"ForceEnableSimd32", i1 false}
!307 = !{!"outputDepth", i1 false}
!308 = !{!"outputStencil", i1 false}
!309 = !{!"outputMask", i1 false}
!310 = !{!"blendToFillEnabled", i1 false}
!311 = !{!"forceEarlyZ", i1 false}
!312 = !{!"hasVersionedLoop", i1 false}
!313 = !{!"forceSingleSourceRTWAfterDualSourceRTW", i1 false}
!314 = !{!"requestCPSizeRelevant", i1 false}
!315 = !{!"requestCPSize", i1 false}
!316 = !{!"texelMaskFastClearMode", !"Disabled"}
!317 = !{!"NumSamples", i8 0}
!318 = !{!"blendOptimizationMode"}
!319 = !{!"colorOutputMask"}
!320 = !{!"ProvokingVertexModeNosIndex", i32 0}
!321 = !{!"ProvokingVertexModeNosPatch", !""}
!322 = !{!"ProvokingVertexModeLast", !"Negative"}
!323 = !{!"VertexAttributesBypass", i1 false}
!324 = !{!"LegacyBaryAssignmentDisableLinear", i1 false}
!325 = !{!"LegacyBaryAssignmentDisableLinearNoPerspective", i1 false}
!326 = !{!"LegacyBaryAssignmentDisableLinearCentroid", i1 false}
!327 = !{!"LegacyBaryAssignmentDisableLinearNoPerspectiveCentroid", i1 false}
!328 = !{!"LegacyBaryAssignmentDisableLinearSample", i1 false}
!329 = !{!"LegacyBaryAssignmentDisableLinearNoPerspectiveSample", i1 false}
!330 = !{!"MeshShaderWAPerPrimitiveUserDataEnable", !"Negative"}
!331 = !{!"meshShaderWAPerPrimitiveUserDataEnablePatchName", !""}
!332 = !{!"generatePatchesForRTWriteSends", i1 false}
!333 = !{!"forceVMask", i1 false}
!334 = !{!"WaDisableVRS", i1 false}
!335 = !{!"RelaxMemoryVisibilityFromPSOrdering", i1 false}
!336 = !{!"WaEnableVMaskUnderNonUnifromCF", i1 false}
!337 = !{!"csInfo", !338, !339, !340, !341, !342, !50, !51, !343, !344, !345, !346, !347, !348, !349, !350, !351, !352, !353, !354, !355, !83, !356, !357, !358, !359, !360, !361, !362}
!338 = !{!"maxWorkGroupSize", i32 0}
!339 = !{!"waveSize", i32 0}
!340 = !{!"ComputeShaderSecondCompile"}
!341 = !{!"forcedSIMDSize", i8 0}
!342 = !{!"forceTotalGRFNum", i32 0}
!343 = !{!"forceSpillCompression", i1 false}
!344 = !{!"allowLowerSimd", i1 false}
!345 = !{!"disableSimd32Slicing", i1 false}
!346 = !{!"disableSplitOnSpill", i1 false}
!347 = !{!"enableNewSpillCostFunction", i1 false}
!348 = !{!"forceVISAPreSched", i1 false}
!349 = !{!"forceUniformBuffer", i1 false}
!350 = !{!"forceUniformSurfaceSampler", i1 false}
!351 = !{!"disableLocalIdOrderOptimizations", i1 false}
!352 = !{!"disableDispatchAlongY", i1 false}
!353 = !{!"neededThreadIdLayout", i1* null}
!354 = !{!"forceTileYWalk", i1 false}
!355 = !{!"atomicBranch", i32 0}
!356 = !{!"disableEarlyOut", i1 false}
!357 = !{!"walkOrderEnabled", i1 false}
!358 = !{!"walkOrderOverride", i32 0}
!359 = !{!"ResForHfPacking"}
!360 = !{!"hasWaveMatrix", i1 false}
!361 = !{!"constantFoldSimdSize", i1 false}
!362 = !{!"isNodeShader", i1 false}
!363 = !{!"msInfo", !364, !365, !366, !367, !368, !369, !370, !371, !372, !373, !374, !322, !320, !375}
!364 = !{!"PrimitiveTopology", i32 3}
!365 = !{!"MaxNumOfPrimitives", i32 0}
!366 = !{!"MaxNumOfVertices", i32 0}
!367 = !{!"MaxNumOfPerPrimitiveOutputs", i32 0}
!368 = !{!"MaxNumOfPerVertexOutputs", i32 0}
!369 = !{!"WorkGroupSize", i32 0}
!370 = !{!"WorkGroupMemorySizeInBytes", i32 0}
!371 = !{!"IndexFormat", i32 6}
!372 = !{!"SubgroupSize", i32 0}
!373 = !{!"VPandRTAIndexAutostripEnable", i1 false}
!374 = !{!"MeshShaderWAPerPrimitiveUserDataEnable", i1 false}
!375 = !{!"numPrimitiveAttributesPatchBaseName", !""}
!376 = !{!"taskInfo", !296, !369, !370, !372}
!377 = !{!"NBarrierCnt", i32 0}
!378 = !{!"rtInfo", !379, !380, !381, !382, !383, !384, !385, !386, !387, !388, !389, !390, !391, !392}
!379 = !{!"RayQueryAllocSizeInBytes", i32 0}
!380 = !{!"NumContinuations", i32 0}
!381 = !{!"RTAsyncStackAddrspace", i32 -1}
!382 = !{!"RTAsyncStackSurfaceStateOffset", i1* null}
!383 = !{!"SWHotZoneAddrspace", i32 -1}
!384 = !{!"SWHotZoneSurfaceStateOffset", i1* null}
!385 = !{!"SWStackAddrspace", i32 -1}
!386 = !{!"SWStackSurfaceStateOffset", i1* null}
!387 = !{!"RTSyncStackAddrspace", i32 -1}
!388 = !{!"RTSyncStackSurfaceStateOffset", i1* null}
!389 = !{!"doSyncDispatchRays", i1 false}
!390 = !{!"MemStyle", !"Xe"}
!391 = !{!"GlobalDataStyle", !"Xe"}
!392 = !{!"NeedsBTD", i1 true}
!393 = !{!"EnableTextureIndirection", i1 false}
!394 = !{!"EnableSamplerIndirection", i1 false}
!395 = !{!"samplerStateStride", i32 0}
!396 = !{!"samplerStateOffset", i32 0}
!397 = !{!"textureStateStride", i32 0}
!398 = !{!"textureStateOffset", i32 0}
!399 = !{!"CurUniqueIndirectIdx", i32 0}
!400 = !{!"inlineDynTextures"}
!401 = !{!"inlineResInfoData"}
!402 = !{!"immConstant", !403, !404, !405}
!403 = !{!"data"}
!404 = !{!"sizes"}
!405 = !{!"zeroIdxs"}
!406 = !{!"stringConstants"}
!407 = !{!"inlineBuffers", !408, !412, !413}
!408 = !{!"inlineBuffersVec[0]", !409, !410, !411}
!409 = !{!"alignment", i32 0}
!410 = !{!"allocSize", i64 0}
!411 = !{!"Buffer"}
!412 = !{!"inlineBuffersVec[1]", !409, !410, !411}
!413 = !{!"inlineBuffersVec[2]", !409, !410, !411}
!414 = !{!"GlobalPointerProgramBinaryInfos"}
!415 = !{!"ConstantPointerProgramBinaryInfos"}
!416 = !{!"GlobalBufferAddressRelocInfo"}
!417 = !{!"ConstantBufferAddressRelocInfo"}
!418 = !{!"forceLscCacheList"}
!419 = !{!"SrvMap"}
!420 = !{!"RootConstantBufferOffsetInBytes"}
!421 = !{!"RasterizerOrderedByteAddressBuffer"}
!422 = !{!"RasterizerOrderedViews"}
!423 = !{!"MinNOSPushConstantSize", i32 0}
!424 = !{!"inlineProgramScopeOffsets"}
!425 = !{!"shaderData", !426}
!426 = !{!"numReplicas", i32 0}
!427 = !{!"URBInfo", !428, !429, !430}
!428 = !{!"has64BVertexHeaderInput", i1 false}
!429 = !{!"has64BVertexHeaderOutput", i1 false}
!430 = !{!"hasVertexHeader", i1 true}
!431 = !{!"m_ForcePullModel", i1 false}
!432 = !{!"UseBindlessImage", i1 false}
!433 = !{!"enableRangeReduce", i1 false}
!434 = !{!"disableNewTrigFuncRangeReduction", i1 false}
!435 = !{!"enableFRemToSRemOpt", i1 false}
!436 = !{!"enableSampleptrToLdmsptrSample0", i1 false}
!437 = !{!"enableSampleLptrToLdmsptrSample0", i1 false}
!438 = !{!"WaForceSIMD32MicropolyRasterize", i1 false}
!439 = !{!"allowMatchMadOptimizationforVS", i1 false}
!440 = !{!"disableMatchMadOptimizationForCS", i1 false}
!441 = !{!"disableMemOptforNegativeOffsetLoads", i1 false}
!442 = !{!"enableThreeWayLoadSpiltOpt", i1 false}
!443 = !{!"statefulResourcesNotAliased", i1 false}
!444 = !{!"disableMixMode", i1 false}
!445 = !{!"genericAccessesResolved", i1 false}
!446 = !{!"disableSeparateSpillPvtScratchSpace", i1 false}
!447 = !{!"disableSeparateScratchWA", i1 false}
!448 = !{!"PrivateMemoryPerFG"}
!449 = !{!"m_OptsToDisable"}
!450 = !{!"capabilities", !451}
!451 = !{!"globalVariableDecorationsINTEL", i1 false}
!452 = !{!"m_ShaderResourceViewMcsMask", !453, !454}
!453 = !{!"m_ShaderResourceViewMcsMaskVec[0]", i64 0}
!454 = !{!"m_ShaderResourceViewMcsMaskVec[1]", i64 0}
!455 = !{!"computedDepthMode", i32 0}
!456 = !{!"isHDCFastClearShader", i1 false}
!457 = !{!"argRegisterReservations", !458}
!458 = !{!"argRegisterReservationsVec[0]", i32 0}
!459 = !{!"SIMD16_SpillThreshold", i8 0}
!460 = !{!"SIMD32_SpillThreshold", i8 0}
!461 = !{!"m_CacheControlOption", !462, !463, !464, !465}
!462 = !{!"LscLoadCacheControlOverride", i8 0}
!463 = !{!"LscStoreCacheControlOverride", i8 0}
!464 = !{!"TgmLoadCacheControlOverride", i8 0}
!465 = !{!"TgmStoreCacheControlOverride", i8 0}
!466 = !{i32 2, i32 0}
!467 = !{!"clang version 14.0.5"}
!468 = !{i32 1, !"wchar_size", i32 4}
!469 = !{i32 1, i32 0, i32 0, i32 0, i32 1, i32 0, i32 0, i32 1, i32 0, i32 0}
!470 = !{!"none", !"none", !"none", !"none", !"none", !"none", !"none", !"none", !"none", !"none"}
!471 = !{!"float*", !"class.sycl::_V1::range", !"class.sycl::_V1::range", !"long", !"class.sycl::_V1::ext::oneapi::bfloat16*", !"class.sycl::_V1::range", !"class.sycl::_V1::range", !"class.sycl::_V1::ext::oneapi::bfloat16*", !"class.sycl::_V1::range", !"class.sycl::_V1::range"}
!472 = !{!"", !"", !"", !"", !"", !"", !"", !"", !"", !""}
!473 = !{!474, !476, !476, !479, !480, !476, !476, !480, !476, !476}
!474 = !{!475}
!475 = !{i32 44, i32 4}
!476 = !{!477, !478}
!477 = !{i32 38, i32 2}
!478 = !{i32 44, i32 8}
!479 = !{}
!480 = !{!481}
!481 = !{i32 44, i32 2}
!482 = !{!483}
!483 = !{i32 4469}
!484 = !{!485, !485, i64 0}
!485 = !{!"int", !486, i64 0}
!486 = !{!"omnipotent char", !487, i64 0}
!487 = !{!"Simple C/C++ TBAA"}
!488 = !{!483, !489}
!489 = !{i32 4470}
