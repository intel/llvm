; ------------------------------------------------
; OCL_asm40d739ce99ad8281_0033_Unify_after_BreakConstantExprPass.ll
; LLVM major version: 14
; ------------------------------------------------
target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v16:16:16-v24:32:32-v32:32:32-v48:64:64-v64:64:64-v96:128:128-v128:128:128-v192:256:256-v256:256:256-v512:512:512-v1024:1024:1024-n8:16:32"
target triple = "spir64-unknown-unknown"

%"class.sycl::_V1::ext::oneapi::bfloat16" = type { i16 }
%"class.sycl::_V1::range" = type { %"class.sycl::_V1::detail::array" }
%"class.sycl::_V1::detail::array" = type { [2 x i64] }

; Function Attrs: nounwind
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
  %.ascast.i3 = addrspacecast float addrspace(1)* %85 to i8 addrspace(4)*
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
  %123 = bitcast i8 addrspace(4)* %.ascast.i3 to i32 addrspace(4)*
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
  %141 = getelementptr inbounds i32, i32 addrspace(4)* %123, i64 %140
  %142 = load i32, i32 addrspace(4)* %141, align 4, !tbaa !455
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
  %166 = phi i32 [ 0, %__builtin_spriv_OpJointMatrixLoadINTEL_Accumulator_RowMajor_32x32_i32_128_global_v8i8_pi32_i32.23.exit ], [ %288, %__builtin_spriv_OpJointMatrixLoadINTEL_PackedB_PackedB_16x32_i16_32_global_v8i8_pi32_i32.25.exit ]
  %167 = icmp ult i32 %166, 2
  br i1 %167, label %168, label %289

168:                                              ; preds = %165
  %169 = shl nuw nsw i32 %166, 4, !spirv.Decorations !459
  %170 = zext i32 %169 to i64
  %171 = getelementptr inbounds %"class.sycl::_V1::ext::oneapi::bfloat16", %"class.sycl::_V1::ext::oneapi::bfloat16" addrspace(1)* %159, i64 %170
  %172 = bitcast <32 x i32>* %12 to i8*
  %.ascast.i5 = addrspacecast %"class.sycl::_V1::ext::oneapi::bfloat16" addrspace(1)* %171 to i8 addrspace(4)*
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
  %210 = bitcast i8 addrspace(4)* %.ascast.i5 to i32 addrspace(4)*
  %211 = addrspacecast i32 addrspace(4)* %210 to i32 addrspace(1)*
  br label %212

212:                                              ; preds = %212, %_Z18get_sub_group_sizev.exit.i6
  %213 = phi i32 [ 0, %_Z18get_sub_group_sizev.exit.i6 ], [ %219, %212 ]
  %214 = zext i32 %213 to i64
  %215 = mul nsw i64 16, %214
  %216 = getelementptr inbounds i32, i32 addrspace(1)* %211, i64 %215
  %217 = call spir_func i32 @__builtin_IB_simd_block_read_1_global(i32 addrspace(1)* noundef %216) #6
  %218 = getelementptr inbounds i32, i32* %209, i64 %214
  store i32 %217, i32* %218, align 4, !tbaa !455
  %219 = add nuw nsw i32 %213, 1
  %220 = icmp ult i32 %213, 31
  br i1 %220, label %212, label %__builtin_spriv_OpJointMatrixLoadINTEL_PackedA_RowMajor_32x16_i16_32_global_v8i8_pi32_i32.24.exit

__builtin_spriv_OpJointMatrixLoadINTEL_PackedA_RowMajor_32x16_i16_32_global_v8i8_pi32_i32.24.exit: ; preds = %212
  %221 = load <32 x i32>, <32 x i32>* %12, align 128
  %222 = shl nuw nsw i64 %170, 6, !spirv.Decorations !459
  %223 = getelementptr %"class.sycl::_V1::ext::oneapi::bfloat16", %"class.sycl::_V1::ext::oneapi::bfloat16" addrspace(1)* %164, i64 %222
  %224 = bitcast <32 x i32>* %11 to i8*
  %.ascast.i7 = addrspacecast %"class.sycl::_V1::ext::oneapi::bfloat16" addrspace(1)* %223 to i8 addrspace(4)*
  %225 = call spir_func i32 @__builtin_IB_get_local_size(i32 noundef 0) #4
  %226 = call spir_func i32 @__builtin_IB_get_local_size(i32 noundef 1) #4
  %227 = mul i32 %226, %225
  %228 = call spir_func i32 @__builtin_IB_get_local_size(i32 noundef 2) #4
  %229 = mul i32 %227, %228
  %230 = call spir_func i32 @__builtin_IB_get_simd_size() #6
  %231 = add i32 %230, -1
  %232 = and i32 %231, %229
  %233 = icmp eq i32 %232, 0
  br i1 %233, label %258, label %234

234:                                              ; preds = %__builtin_spriv_OpJointMatrixLoadINTEL_PackedA_RowMajor_32x16_i16_32_global_v8i8_pi32_i32.24.exit
  %235 = call spir_func i32 @__builtin_IB_get_local_id_z() #4
  %236 = call spir_func i32 @__builtin_IB_get_local_size(i32 noundef 1) #4
  %237 = mul i32 %236, %235
  %238 = call spir_func i32 @__builtin_IB_get_local_id_y() #4
  %239 = add i32 %237, %238
  %240 = call spir_func i32 @__builtin_IB_get_local_size(i32 noundef 0) #4
  %241 = mul i32 %239, %240
  %242 = call spir_func i32 @__builtin_IB_get_local_id_x() #4
  %243 = add i32 %241, %242
  %244 = call spir_func i32 @__builtin_IB_get_simd_size() #6
  %245 = udiv i32 %243, %244
  %246 = call spir_func i32 @__builtin_IB_get_local_size(i32 noundef 0) #4
  %247 = call spir_func i32 @__builtin_IB_get_local_size(i32 noundef 1) #4
  %248 = mul i32 %247, %246
  %249 = call spir_func i32 @__builtin_IB_get_local_size(i32 noundef 2) #4
  %250 = mul i32 %248, %249
  %251 = call spir_func i32 @__builtin_IB_get_simd_size() #6
  %252 = add i32 %250, -1
  %253 = add i32 %252, %251
  %254 = call spir_func i32 @__builtin_IB_get_simd_size() #6
  %255 = udiv i32 %253, %254
  %256 = add i32 %255, -1
  %257 = icmp ult i32 %245, %256
  br i1 %257, label %258, label %_Z18get_sub_group_sizev.exit.i8

258:                                              ; preds = %234, %__builtin_spriv_OpJointMatrixLoadINTEL_PackedA_RowMajor_32x16_i16_32_global_v8i8_pi32_i32.24.exit
  %259 = call spir_func i32 @__builtin_IB_get_simd_size() #6
  br label %_Z18get_sub_group_sizev.exit.i8

_Z18get_sub_group_sizev.exit.i8:                  ; preds = %258, %234
  %260 = phi i32 [ %259, %258 ], [ %232, %234 ]
  %261 = bitcast i8 addrspace(4)* %.ascast.i7 to i32 addrspace(4)*
  %262 = call spir_func i32 @__builtin_IB_get_simd_id() #6
  %263 = sdiv i32 %260, 32
  %264 = bitcast i8* %224 to i32*
  %265 = freeze i32 %262
  %266 = sdiv i32 %265, 32
  %267 = mul i32 %266, 32
  %268 = sub i32 %265, %267
  %269 = sext i32 %268 to i64
  br label %270

270:                                              ; preds = %281, %_Z18get_sub_group_sizev.exit.i8
  %271 = phi i32 [ 0, %_Z18get_sub_group_sizev.exit.i8 ], [ %285, %281 ]
  %272 = mul nsw i32 %271, %263
  %273 = add nsw i32 %272, %266
  %274 = icmp slt i32 %273, 8
  br i1 %274, label %275, label %281

275:                                              ; preds = %270
  %276 = sext i32 %273 to i64
  %277 = mul nsw i64 64, %276
  %278 = add nsw i64 %277, %269
  %279 = getelementptr inbounds i32, i32 addrspace(4)* %261, i64 %278
  %280 = load i32, i32 addrspace(4)* %279, align 4, !tbaa !455
  br label %281

281:                                              ; preds = %275, %270
  %282 = phi i32 [ %280, %275 ], [ 0, %270 ]
  %283 = zext i32 %271 to i64
  %284 = getelementptr inbounds i32, i32* %264, i64 %283
  store i32 %282, i32* %284, align 4, !tbaa !455
  %285 = add nuw nsw i32 %271, 1
  %286 = icmp ult i32 %271, 31
  br i1 %286, label %270, label %__builtin_spriv_OpJointMatrixLoadINTEL_PackedB_PackedB_16x32_i16_32_global_v8i8_pi32_i32.25.exit

__builtin_spriv_OpJointMatrixLoadINTEL_PackedB_PackedB_16x32_i16_32_global_v8i8_pi32_i32.25.exit: ; preds = %281
  %287 = load <32 x i32>, <32 x i32>* %11, align 128
  %288 = add nuw nsw i32 %166, 1, !spirv.Decorations !459
  br label %165

289:                                              ; preds = %165
  store [2 x <64 x float>] %154, [2 x <64 x float>]* %13, align 256
  %290 = bitcast [2 x <64 x float>]* %13 to i8*
  %.ascast.i = addrspacecast float addrspace(1)* %85 to i8 addrspace(4)*
  %291 = call spir_func i32 @__builtin_IB_get_local_size(i32 noundef 0) #4
  %292 = call spir_func i32 @__builtin_IB_get_local_size(i32 noundef 1) #4
  %293 = mul i32 %292, %291
  %294 = call spir_func i32 @__builtin_IB_get_local_size(i32 noundef 2) #4
  %295 = mul i32 %293, %294
  %296 = call spir_func i32 @__builtin_IB_get_simd_size() #6
  %297 = add i32 %296, -1
  %298 = and i32 %297, %295
  %299 = icmp eq i32 %298, 0
  br i1 %299, label %324, label %300

300:                                              ; preds = %289
  %301 = call spir_func i32 @__builtin_IB_get_local_id_z() #4
  %302 = call spir_func i32 @__builtin_IB_get_local_size(i32 noundef 1) #4
  %303 = mul i32 %302, %301
  %304 = call spir_func i32 @__builtin_IB_get_local_id_y() #4
  %305 = add i32 %303, %304
  %306 = call spir_func i32 @__builtin_IB_get_local_size(i32 noundef 0) #4
  %307 = mul i32 %305, %306
  %308 = call spir_func i32 @__builtin_IB_get_local_id_x() #4
  %309 = add i32 %307, %308
  %310 = call spir_func i32 @__builtin_IB_get_simd_size() #6
  %311 = udiv i32 %309, %310
  %312 = call spir_func i32 @__builtin_IB_get_local_size(i32 noundef 0) #4
  %313 = call spir_func i32 @__builtin_IB_get_local_size(i32 noundef 1) #4
  %314 = mul i32 %313, %312
  %315 = call spir_func i32 @__builtin_IB_get_local_size(i32 noundef 2) #4
  %316 = mul i32 %314, %315
  %317 = call spir_func i32 @__builtin_IB_get_simd_size() #6
  %318 = add i32 %316, -1
  %319 = add i32 %318, %317
  %320 = call spir_func i32 @__builtin_IB_get_simd_size() #6
  %321 = udiv i32 %319, %320
  %322 = add i32 %321, -1
  %323 = icmp ult i32 %311, %322
  br i1 %323, label %324, label %_Z18get_sub_group_sizev.exit.i

324:                                              ; preds = %300, %289
  %325 = call spir_func i32 @__builtin_IB_get_simd_size() #6
  br label %_Z18get_sub_group_sizev.exit.i

_Z18get_sub_group_sizev.exit.i:                   ; preds = %324, %300
  %326 = phi i32 [ %325, %324 ], [ %298, %300 ]
  %327 = bitcast i8 addrspace(4)* %.ascast.i to i32 addrspace(4)*
  %328 = call spir_func i32 @__builtin_IB_get_simd_id() #6
  %329 = sdiv i32 %326, 32
  %330 = bitcast i8* %290 to i32*
  %331 = freeze i32 %328
  %332 = sdiv i32 %331, 32
  %333 = mul i32 %332, 32
  %334 = sub i32 %331, %333
  %335 = sext i32 %334 to i64
  br label %336

336:                                              ; preds = %349, %_Z18get_sub_group_sizev.exit.i
  %337 = phi i32 [ 0, %_Z18get_sub_group_sizev.exit.i ], [ %350, %349 ]
  %338 = mul nsw i32 %337, %329
  %339 = add nsw i32 %338, %332
  %340 = icmp slt i32 %339, 32
  br i1 %340, label %341, label %349

341:                                              ; preds = %336
  %342 = zext i32 %337 to i64
  %343 = getelementptr inbounds i32, i32* %330, i64 %342
  %344 = load i32, i32* %343, align 4, !tbaa !455
  %345 = sext i32 %339 to i64
  %346 = mul nsw i64 %345, 64
  %347 = add nsw i64 %346, %335
  %348 = getelementptr inbounds i32, i32 addrspace(4)* %327, i64 %347
  store i32 %344, i32 addrspace(4)* %348, align 4, !tbaa !455
  br label %349

349:                                              ; preds = %341, %336
  %350 = add nuw nsw i32 %337, 1
  %351 = icmp ult i32 %337, 127
  br i1 %351, label %336, label %__builtin_spriv_OpJointMatrixStoreINTEL_Accumulator_RowMajor_32x32_i32_128_global_pi64_v8i8.26.exit

__builtin_spriv_OpJointMatrixStoreINTEL_Accumulator_RowMajor_32x32_i32_128_global_pi64_v8i8.26.exit: ; preds = %349
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

attributes #0 = { nounwind "less-precise-fpmad"="true" }
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
!IGCMetadata = !{!7}
!opencl.ocl.version = !{!437, !437, !437, !437, !437}
!opencl.spir.version = !{!437, !437, !437, !437, !437}
!llvm.ident = !{!438, !438, !438, !438, !438}
!llvm.module.flags = !{!439}

!0 = !{i32 2, i32 2}
!1 = !{i32 4, i32 100000}
!2 = !{i16 6, i16 14}
!3 = !{void (float addrspace(1)*, %"class.sycl::_V1::range"*, %"class.sycl::_V1::range"*, i64, %"class.sycl::_V1::ext::oneapi::bfloat16" addrspace(1)*, %"class.sycl::_V1::range"*, %"class.sycl::_V1::range"*, %"class.sycl::_V1::ext::oneapi::bfloat16" addrspace(1)*, %"class.sycl::_V1::range"*, %"class.sycl::_V1::range"*)* @_ZTS7imatrixIfLm32ELm32ELm16EE, !4}
!4 = !{!5, !6}
!5 = !{!"function_type", i32 0}
!6 = !{!"sub_group_size", i32 8}
!7 = !{!"ModuleMD", !8, !9, !106, !232, !263, !264, !268, !271, !272, !273, !308, !334, !347, !348, !349, !364, !365, !366, !367, !368, !369, !370, !371, !372, !373, !377, !378, !385, !386, !387, !388, !389, !390, !391, !392, !393, !394, !395, !396, !398, !402, !403, !404, !405, !406, !407, !408, !409, !410, !411, !412, !413, !414, !415, !416, !417, !418, !156, !419, !420, !421, !423, !426, !427, !428, !430, !431, !432}
!8 = !{!"isPrecise", i1 false}
!9 = !{!"compOpt", !10, !11, !12, !13, !14, !15, !16, !17, !18, !19, !20, !21, !22, !23, !24, !25, !26, !27, !28, !29, !30, !31, !32, !33, !34, !35, !36, !37, !38, !39, !40, !41, !42, !43, !44, !45, !46, !47, !48, !49, !50, !51, !52, !53, !54, !55, !56, !57, !58, !59, !60, !61, !62, !63, !64, !65, !66, !67, !68, !69, !70, !71, !72, !73, !74, !75, !76, !77, !78, !79, !80, !81, !82, !83, !84, !85, !86, !87, !88, !89, !90, !91, !92, !93, !94, !95, !96, !97, !98, !99, !100, !101, !102, !103, !104, !105}
!10 = !{!"DenormsAreZero", i1 false}
!11 = !{!"BFTFDenormsAreZero", i1 false}
!12 = !{!"CorrectlyRoundedDivSqrt", i1 false}
!13 = !{!"OptDisable", i1 false}
!14 = !{!"MadEnable", i1 true}
!15 = !{!"NoSignedZeros", i1 false}
!16 = !{!"NoNaNs", i1 false}
!17 = !{!"FloatRoundingMode", i32 0}
!18 = !{!"FloatCvtIntRoundingMode", i32 3}
!19 = !{!"LoadCacheDefault", i32 4}
!20 = !{!"StoreCacheDefault", i32 7}
!21 = !{!"VISAPreSchedRPThreshold", i32 0}
!22 = !{!"SetLoopUnrollThreshold", i32 0}
!23 = !{!"UnsafeMathOptimizations", i1 false}
!24 = !{!"disableCustomUnsafeOpts", i1 false}
!25 = !{!"disableReducePow", i1 false}
!26 = !{!"disableSqrtOpt", i1 false}
!27 = !{!"FiniteMathOnly", i1 false}
!28 = !{!"FastRelaxedMath", i1 false}
!29 = !{!"DashGSpecified", i1 false}
!30 = !{!"FastCompilation", i1 false}
!31 = !{!"UseScratchSpacePrivateMemory", i1 true}
!32 = !{!"RelaxedBuiltins", i1 false}
!33 = !{!"SubgroupIndependentForwardProgressRequired", i1 true}
!34 = !{!"GreaterThan2GBBufferRequired", i1 true}
!35 = !{!"GreaterThan4GBBufferRequired", i1 false}
!36 = !{!"DisableA64WA", i1 false}
!37 = !{!"ForceEnableA64WA", i1 false}
!38 = !{!"PushConstantsEnable", i1 true}
!39 = !{!"HasPositivePointerOffset", i1 false}
!40 = !{!"HasBufferOffsetArg", i1 true}
!41 = !{!"BufferOffsetArgOptional", i1 true}
!42 = !{!"replaceGlobalOffsetsByZero", i1 false}
!43 = !{!"forcePixelShaderSIMDMode", i32 0}
!44 = !{!"ForceGeomFFShaderSIMDMode", i32 0}
!45 = !{!"pixelShaderDoNotAbortOnSpill", i1 false}
!46 = !{!"UniformWGS", i1 false}
!47 = !{!"disableVertexComponentPacking", i1 false}
!48 = !{!"disablePartialVertexComponentPacking", i1 false}
!49 = !{!"PreferBindlessImages", i1 false}
!50 = !{!"UseBindlessMode", i1 false}
!51 = !{!"UseLegacyBindlessMode", i1 true}
!52 = !{!"disableMathRefactoring", i1 false}
!53 = !{!"atomicBranch", i1 false}
!54 = !{!"spillCompression", i1 false}
!55 = !{!"DisableEarlyOut", i1 false}
!56 = !{!"ForceInt32DivRemEmu", i1 false}
!57 = !{!"ForceInt32DivRemEmuSP", i1 false}
!58 = !{!"WaveIntrinsicUsed", i1 false}
!59 = !{!"DisableMultiPolyPS", i1 false}
!60 = !{!"NeedTexture3DLODWA", i1 false}
!61 = !{!"DisableFastestSingleCSSIMD", i1 false}
!62 = !{!"DisableFastestLinearScan", i1 false}
!63 = !{!"UseStatelessforPrivateMemory", i1 false}
!64 = !{!"EnableTakeGlobalAddress", i1 false}
!65 = !{!"IsLibraryCompilation", i1 false}
!66 = !{!"LibraryCompileSIMDSize", i32 0}
!67 = !{!"FastVISACompile", i1 false}
!68 = !{!"MatchSinCosPi", i1 false}
!69 = !{!"ExcludeIRFromZEBinary", i1 false}
!70 = !{!"EmitZeBinVISASections", i1 false}
!71 = !{!"FP64GenEmulationEnabled", i1 false}
!72 = !{!"FP64GenConvEmulationEnabled", i1 false}
!73 = !{!"allowDisableRematforCS", i1 false}
!74 = !{!"DisableIncSpillCostAllAddrTaken", i1 false}
!75 = !{!"DisableCPSOmaskWA", i1 false}
!76 = !{!"DisableFastestGopt", i1 false}
!77 = !{!"WaForceHalfPromotionComputeShader", i1 false}
!78 = !{!"WaForceHalfPromotionPixelVertexShader", i1 false}
!79 = !{!"DisableConstantCoalescing", i1 false}
!80 = !{!"EnableUndefAlphaOutputAsRed", i1 true}
!81 = !{!"WaEnableALTModeVisaWA", i1 false}
!82 = !{!"WaEnableAtomicWaveFusion", i1 false}
!83 = !{!"WaEnableAtomicWaveFusionNonNullResource", i1 false}
!84 = !{!"WaEnableAtomicWaveFusionStateless", i1 false}
!85 = !{!"WaEnableAtomicWaveFusionTyped", i1 false}
!86 = !{!"ForceCBThroughSampler3D", i1 false}
!87 = !{!"WaStoreRawVectorToTypedWrite", i1 false}
!88 = !{!"WaLoadRawVectorToTypedRead", i1 false}
!89 = !{!"WaZeroSLMBeforeUse", i1 false}
!90 = !{!"WaFlagGroupTypedUAVGloballyCoherent", i1 false}
!91 = !{!"NewSpillCostFunction", i1 false}
!92 = !{!"EnableVRT", i1 false}
!93 = !{!"ForceLargeGRFNum4RQ", i1 false}
!94 = !{!"Enable2xGRFRetry", i1 false}
!95 = !{!"Detect2xGRFCandidate", i1 false}
!96 = !{!"EnableURBWritesMerging", i1 true}
!97 = !{!"DisableEUFusion", i1 false}
!98 = !{!"DisableFDivToFMulInvOpt", i1 false}
!99 = !{!"initializePhiSampleSourceWA", i1 false}
!100 = !{!"WaDisableSubspanUseNoMaskForCB", i1 false}
!101 = !{!"DisableLoosenSimd32Occu", i1 false}
!102 = !{!"FastestS1Options", i32 0}
!103 = !{!"DisableFastestForWaveIntrinsicsCS", i1 false}
!104 = !{!"ForceLinearWalkOnLinearUAV", i1 false}
!105 = !{!"DisableLscSamplerRouting", i1 false}
!106 = !{!"FuncMD", !107, !108}
!107 = !{!"FuncMDMap[0]", void (float addrspace(1)*, %"class.sycl::_V1::range"*, %"class.sycl::_V1::range"*, i64, %"class.sycl::_V1::ext::oneapi::bfloat16" addrspace(1)*, %"class.sycl::_V1::range"*, %"class.sycl::_V1::range"*, %"class.sycl::_V1::ext::oneapi::bfloat16" addrspace(1)*, %"class.sycl::_V1::range"*, %"class.sycl::_V1::range"*)* @_ZTS7imatrixIfLm32ELm32ELm16EE}
!108 = !{!"FuncMDValue[0]", !109, !110, !114, !115, !116, !117, !118, !119, !120, !142, !148, !149, !150, !151, !152, !153, !154, !155, !156, !157, !158, !159, !160, !161, !162, !163, !164, !175, !186, !197, !208, !219, !230, !231}
!109 = !{!"localOffsets"}
!110 = !{!"workGroupWalkOrder", !111, !112, !113}
!111 = !{!"dim0", i32 0}
!112 = !{!"dim1", i32 1}
!113 = !{!"dim2", i32 2}
!114 = !{!"funcArgs"}
!115 = !{!"functionType", !"KernelFunction"}
!116 = !{!"inlineDynConstants"}
!117 = !{!"inlineDynRootConstant"}
!118 = !{!"inlineDynConstantDescTable"}
!119 = !{!"m_pInterestingConstants"}
!120 = !{!"rtInfo", !121, !122, !123, !124, !125, !126, !127, !128, !129, !130, !131, !132, !133, !134, !135, !136, !140, !141, !92}
!121 = !{!"callableShaderType", !"NumberOfCallableShaderTypes"}
!122 = !{!"isContinuation", i1 false}
!123 = !{!"hasTraceRayPayload", i1 false}
!124 = !{!"hasHitAttributes", i1 false}
!125 = !{!"hasCallableData", i1 false}
!126 = !{!"ShaderStackSize", i32 0}
!127 = !{!"ShaderHash", i64 0}
!128 = !{!"ShaderName", !""}
!129 = !{!"ParentName", !""}
!130 = !{!"SlotNum", i1* null}
!131 = !{!"NOSSize", i32 0}
!132 = !{!"globalRootSignatureSize", i32 0}
!133 = !{!"Entries"}
!134 = !{!"SpillUnions"}
!135 = !{!"CustomHitAttrSizeInBytes", i32 0}
!136 = !{!"Types", !137, !138, !139}
!137 = !{!"FrameStartTys"}
!138 = !{!"ArgumentTys"}
!139 = !{!"FullFrameTys"}
!140 = !{!"Aliases"}
!141 = !{!"NumGRF", i32 0}
!142 = !{!"resAllocMD", !143, !144, !145, !146, !147}
!143 = !{!"uavsNumType", i32 0}
!144 = !{!"srvsNumType", i32 0}
!145 = !{!"samplersNumType", i32 0}
!146 = !{!"argAllocMDList"}
!147 = !{!"inlineSamplersMD"}
!148 = !{!"maxByteOffsets"}
!149 = !{!"IsInitializer", i1 false}
!150 = !{!"IsFinalizer", i1 false}
!151 = !{!"CompiledSubGroupsNumber", i32 0}
!152 = !{!"hasInlineVmeSamplers", i1 false}
!153 = !{!"localSize", i32 0}
!154 = !{!"localIDPresent", i1 false}
!155 = !{!"groupIDPresent", i1 false}
!156 = !{!"privateMemoryPerWI", i32 0}
!157 = !{!"prevFPOffset", i32 0}
!158 = !{!"globalIDPresent", i1 false}
!159 = !{!"hasSyncRTCalls", i1 false}
!160 = !{!"hasNonKernelArgLoad", i1 false}
!161 = !{!"hasNonKernelArgStore", i1 false}
!162 = !{!"hasNonKernelArgAtomic", i1 false}
!163 = !{!"UserAnnotations"}
!164 = !{!"m_OpenCLArgAddressSpaces", !165, !166, !167, !168, !169, !170, !171, !172, !173, !174}
!165 = !{!"m_OpenCLArgAddressSpacesVec[0]", i32 1}
!166 = !{!"m_OpenCLArgAddressSpacesVec[1]", i32 0}
!167 = !{!"m_OpenCLArgAddressSpacesVec[2]", i32 0}
!168 = !{!"m_OpenCLArgAddressSpacesVec[3]", i32 0}
!169 = !{!"m_OpenCLArgAddressSpacesVec[4]", i32 1}
!170 = !{!"m_OpenCLArgAddressSpacesVec[5]", i32 0}
!171 = !{!"m_OpenCLArgAddressSpacesVec[6]", i32 0}
!172 = !{!"m_OpenCLArgAddressSpacesVec[7]", i32 1}
!173 = !{!"m_OpenCLArgAddressSpacesVec[8]", i32 0}
!174 = !{!"m_OpenCLArgAddressSpacesVec[9]", i32 0}
!175 = !{!"m_OpenCLArgAccessQualifiers", !176, !177, !178, !179, !180, !181, !182, !183, !184, !185}
!176 = !{!"m_OpenCLArgAccessQualifiersVec[0]", !"none"}
!177 = !{!"m_OpenCLArgAccessQualifiersVec[1]", !"none"}
!178 = !{!"m_OpenCLArgAccessQualifiersVec[2]", !"none"}
!179 = !{!"m_OpenCLArgAccessQualifiersVec[3]", !"none"}
!180 = !{!"m_OpenCLArgAccessQualifiersVec[4]", !"none"}
!181 = !{!"m_OpenCLArgAccessQualifiersVec[5]", !"none"}
!182 = !{!"m_OpenCLArgAccessQualifiersVec[6]", !"none"}
!183 = !{!"m_OpenCLArgAccessQualifiersVec[7]", !"none"}
!184 = !{!"m_OpenCLArgAccessQualifiersVec[8]", !"none"}
!185 = !{!"m_OpenCLArgAccessQualifiersVec[9]", !"none"}
!186 = !{!"m_OpenCLArgTypes", !187, !188, !189, !190, !191, !192, !193, !194, !195, !196}
!187 = !{!"m_OpenCLArgTypesVec[0]", !"float*"}
!188 = !{!"m_OpenCLArgTypesVec[1]", !"class.sycl::_V1::range"}
!189 = !{!"m_OpenCLArgTypesVec[2]", !"class.sycl::_V1::range"}
!190 = !{!"m_OpenCLArgTypesVec[3]", !"long"}
!191 = !{!"m_OpenCLArgTypesVec[4]", !"class.sycl::_V1::ext::oneapi::bfloat16*"}
!192 = !{!"m_OpenCLArgTypesVec[5]", !"class.sycl::_V1::range"}
!193 = !{!"m_OpenCLArgTypesVec[6]", !"class.sycl::_V1::range"}
!194 = !{!"m_OpenCLArgTypesVec[7]", !"class.sycl::_V1::ext::oneapi::bfloat16*"}
!195 = !{!"m_OpenCLArgTypesVec[8]", !"class.sycl::_V1::range"}
!196 = !{!"m_OpenCLArgTypesVec[9]", !"class.sycl::_V1::range"}
!197 = !{!"m_OpenCLArgBaseTypes", !198, !199, !200, !201, !202, !203, !204, !205, !206, !207}
!198 = !{!"m_OpenCLArgBaseTypesVec[0]", !"float*"}
!199 = !{!"m_OpenCLArgBaseTypesVec[1]", !"class.sycl::_V1::range"}
!200 = !{!"m_OpenCLArgBaseTypesVec[2]", !"class.sycl::_V1::range"}
!201 = !{!"m_OpenCLArgBaseTypesVec[3]", !"long"}
!202 = !{!"m_OpenCLArgBaseTypesVec[4]", !"class.sycl::_V1::ext::oneapi::bfloat16*"}
!203 = !{!"m_OpenCLArgBaseTypesVec[5]", !"class.sycl::_V1::range"}
!204 = !{!"m_OpenCLArgBaseTypesVec[6]", !"class.sycl::_V1::range"}
!205 = !{!"m_OpenCLArgBaseTypesVec[7]", !"class.sycl::_V1::ext::oneapi::bfloat16*"}
!206 = !{!"m_OpenCLArgBaseTypesVec[8]", !"class.sycl::_V1::range"}
!207 = !{!"m_OpenCLArgBaseTypesVec[9]", !"class.sycl::_V1::range"}
!208 = !{!"m_OpenCLArgTypeQualifiers", !209, !210, !211, !212, !213, !214, !215, !216, !217, !218}
!209 = !{!"m_OpenCLArgTypeQualifiersVec[0]", !""}
!210 = !{!"m_OpenCLArgTypeQualifiersVec[1]", !""}
!211 = !{!"m_OpenCLArgTypeQualifiersVec[2]", !""}
!212 = !{!"m_OpenCLArgTypeQualifiersVec[3]", !""}
!213 = !{!"m_OpenCLArgTypeQualifiersVec[4]", !""}
!214 = !{!"m_OpenCLArgTypeQualifiersVec[5]", !""}
!215 = !{!"m_OpenCLArgTypeQualifiersVec[6]", !""}
!216 = !{!"m_OpenCLArgTypeQualifiersVec[7]", !""}
!217 = !{!"m_OpenCLArgTypeQualifiersVec[8]", !""}
!218 = !{!"m_OpenCLArgTypeQualifiersVec[9]", !""}
!219 = !{!"m_OpenCLArgNames", !220, !221, !222, !223, !224, !225, !226, !227, !228, !229}
!220 = !{!"m_OpenCLArgNamesVec[0]", !""}
!221 = !{!"m_OpenCLArgNamesVec[1]", !""}
!222 = !{!"m_OpenCLArgNamesVec[2]", !""}
!223 = !{!"m_OpenCLArgNamesVec[3]", !""}
!224 = !{!"m_OpenCLArgNamesVec[4]", !""}
!225 = !{!"m_OpenCLArgNamesVec[5]", !""}
!226 = !{!"m_OpenCLArgNamesVec[6]", !""}
!227 = !{!"m_OpenCLArgNamesVec[7]", !""}
!228 = !{!"m_OpenCLArgNamesVec[8]", !""}
!229 = !{!"m_OpenCLArgNamesVec[9]", !""}
!230 = !{!"m_OpenCLArgScalarAsPointers"}
!231 = !{!"m_OptsToDisablePerFunc"}
!232 = !{!"pushInfo", !233, !234, !235, !239, !240, !241, !242, !243, !244, !245, !246, !259, !260, !261, !262}
!233 = !{!"pushableAddresses"}
!234 = !{!"bindlessPushInfo"}
!235 = !{!"dynamicBufferInfo", !236, !237, !238}
!236 = !{!"firstIndex", i32 0}
!237 = !{!"numOffsets", i32 0}
!238 = !{!"forceDisabled", i1 false}
!239 = !{!"MaxNumberOfPushedBuffers", i32 0}
!240 = !{!"inlineConstantBufferSlot", i32 -1}
!241 = !{!"inlineConstantBufferOffset", i32 -1}
!242 = !{!"inlineConstantBufferGRFOffset", i32 -1}
!243 = !{!"constants"}
!244 = !{!"inputs"}
!245 = !{!"constantReg"}
!246 = !{!"simplePushInfoArr", !247, !256, !257, !258}
!247 = !{!"simplePushInfoArrVec[0]", !248, !249, !250, !251, !252, !253, !254, !255}
!248 = !{!"cbIdx", i32 0}
!249 = !{!"pushableAddressGrfOffset", i32 -1}
!250 = !{!"pushableOffsetGrfOffset", i32 -1}
!251 = !{!"offset", i32 0}
!252 = !{!"size", i32 0}
!253 = !{!"isStateless", i1 false}
!254 = !{!"isBindless", i1 false}
!255 = !{!"simplePushLoads"}
!256 = !{!"simplePushInfoArrVec[1]", !248, !249, !250, !251, !252, !253, !254, !255}
!257 = !{!"simplePushInfoArrVec[2]", !248, !249, !250, !251, !252, !253, !254, !255}
!258 = !{!"simplePushInfoArrVec[3]", !248, !249, !250, !251, !252, !253, !254, !255}
!259 = !{!"simplePushBufferUsed", i32 0}
!260 = !{!"pushAnalysisWIInfos"}
!261 = !{!"inlineRTGlobalPtrOffset", i32 0}
!262 = !{!"rtSyncSurfPtrOffset", i32 0}
!263 = !{!"WaEnableICBPromotion", i1 false}
!264 = !{!"vsInfo", !265, !266, !267}
!265 = !{!"DrawIndirectBufferIndex", i32 -1}
!266 = !{!"vertexReordering", i32 -1}
!267 = !{!"MaxNumOfOutputs", i32 0}
!268 = !{!"hsInfo", !269, !270}
!269 = !{!"numPatchAttributesPatchBaseName", !""}
!270 = !{!"numVertexAttributesPatchBaseName", !""}
!271 = !{!"dsInfo", !267}
!272 = !{!"gsInfo", !267}
!273 = !{!"psInfo", !274, !275, !276, !277, !278, !279, !280, !281, !282, !283, !284, !285, !286, !287, !288, !289, !290, !291, !292, !293, !294, !295, !296, !297, !298, !299, !300, !301, !302, !303, !304, !305, !306, !307}
!274 = !{!"BlendStateDisabledMask", i8 0}
!275 = !{!"SkipSrc0Alpha", i1 false}
!276 = !{!"DualSourceBlendingDisabled", i1 false}
!277 = !{!"ForceEnableSimd32", i1 false}
!278 = !{!"outputDepth", i1 false}
!279 = !{!"outputStencil", i1 false}
!280 = !{!"outputMask", i1 false}
!281 = !{!"blendToFillEnabled", i1 false}
!282 = !{!"forceEarlyZ", i1 false}
!283 = !{!"hasVersionedLoop", i1 false}
!284 = !{!"forceSingleSourceRTWAfterDualSourceRTW", i1 false}
!285 = !{!"requestCPSizeRelevant", i1 false}
!286 = !{!"requestCPSize", i1 false}
!287 = !{!"texelMaskFastClearMode", !"Disabled"}
!288 = !{!"NumSamples", i8 0}
!289 = !{!"blendOptimizationMode"}
!290 = !{!"colorOutputMask"}
!291 = !{!"ProvokingVertexModeNosIndex", i32 0}
!292 = !{!"ProvokingVertexModeNosPatch", !""}
!293 = !{!"ProvokingVertexModeLast", !"Negative"}
!294 = !{!"VertexAttributesBypass", i1 false}
!295 = !{!"LegacyBaryAssignmentDisableLinear", i1 false}
!296 = !{!"LegacyBaryAssignmentDisableLinearNoPerspective", i1 false}
!297 = !{!"LegacyBaryAssignmentDisableLinearCentroid", i1 false}
!298 = !{!"LegacyBaryAssignmentDisableLinearNoPerspectiveCentroid", i1 false}
!299 = !{!"LegacyBaryAssignmentDisableLinearSample", i1 false}
!300 = !{!"LegacyBaryAssignmentDisableLinearNoPerspectiveSample", i1 false}
!301 = !{!"MeshShaderWAPerPrimitiveUserDataEnable", !"Negative"}
!302 = !{!"meshShaderWAPerPrimitiveUserDataEnablePatchName", !""}
!303 = !{!"generatePatchesForRTWriteSends", i1 false}
!304 = !{!"forceVMask", i1 false}
!305 = !{!"WaDisableVRS", i1 false}
!306 = !{!"RelaxMemoryVisibilityFromPSOrdering", i1 false}
!307 = !{!"WaEnableVMaskUnderNonUnifromCF", i1 false}
!308 = !{!"csInfo", !309, !310, !311, !312, !313, !21, !22, !314, !315, !316, !317, !318, !319, !320, !321, !322, !323, !324, !325, !326, !54, !327, !328, !329, !330, !331, !332, !333}
!309 = !{!"maxWorkGroupSize", i32 0}
!310 = !{!"waveSize", i32 0}
!311 = !{!"ComputeShaderSecondCompile"}
!312 = !{!"forcedSIMDSize", i8 0}
!313 = !{!"forceTotalGRFNum", i32 0}
!314 = !{!"forceSpillCompression", i1 false}
!315 = !{!"allowLowerSimd", i1 false}
!316 = !{!"disableSimd32Slicing", i1 false}
!317 = !{!"disableSplitOnSpill", i1 false}
!318 = !{!"enableNewSpillCostFunction", i1 false}
!319 = !{!"forceVISAPreSched", i1 false}
!320 = !{!"forceUniformBuffer", i1 false}
!321 = !{!"forceUniformSurfaceSampler", i1 false}
!322 = !{!"disableLocalIdOrderOptimizations", i1 false}
!323 = !{!"disableDispatchAlongY", i1 false}
!324 = !{!"neededThreadIdLayout", i1* null}
!325 = !{!"forceTileYWalk", i1 false}
!326 = !{!"atomicBranch", i32 0}
!327 = !{!"disableEarlyOut", i1 false}
!328 = !{!"walkOrderEnabled", i1 false}
!329 = !{!"walkOrderOverride", i32 0}
!330 = !{!"ResForHfPacking"}
!331 = !{!"hasWaveMatrix", i1 false}
!332 = !{!"constantFoldSimdSize", i1 false}
!333 = !{!"isNodeShader", i1 false}
!334 = !{!"msInfo", !335, !336, !337, !338, !339, !340, !341, !342, !343, !344, !345, !293, !291, !346}
!335 = !{!"PrimitiveTopology", i32 3}
!336 = !{!"MaxNumOfPrimitives", i32 0}
!337 = !{!"MaxNumOfVertices", i32 0}
!338 = !{!"MaxNumOfPerPrimitiveOutputs", i32 0}
!339 = !{!"MaxNumOfPerVertexOutputs", i32 0}
!340 = !{!"WorkGroupSize", i32 0}
!341 = !{!"WorkGroupMemorySizeInBytes", i32 0}
!342 = !{!"IndexFormat", i32 6}
!343 = !{!"SubgroupSize", i32 0}
!344 = !{!"VPandRTAIndexAutostripEnable", i1 false}
!345 = !{!"MeshShaderWAPerPrimitiveUserDataEnable", i1 false}
!346 = !{!"numPrimitiveAttributesPatchBaseName", !""}
!347 = !{!"taskInfo", !267, !340, !341, !343}
!348 = !{!"NBarrierCnt", i32 0}
!349 = !{!"rtInfo", !350, !351, !352, !353, !354, !355, !356, !357, !358, !359, !360, !361, !362, !363}
!350 = !{!"RayQueryAllocSizeInBytes", i32 0}
!351 = !{!"NumContinuations", i32 0}
!352 = !{!"RTAsyncStackAddrspace", i32 -1}
!353 = !{!"RTAsyncStackSurfaceStateOffset", i1* null}
!354 = !{!"SWHotZoneAddrspace", i32 -1}
!355 = !{!"SWHotZoneSurfaceStateOffset", i1* null}
!356 = !{!"SWStackAddrspace", i32 -1}
!357 = !{!"SWStackSurfaceStateOffset", i1* null}
!358 = !{!"RTSyncStackAddrspace", i32 -1}
!359 = !{!"RTSyncStackSurfaceStateOffset", i1* null}
!360 = !{!"doSyncDispatchRays", i1 false}
!361 = !{!"MemStyle", !"Xe"}
!362 = !{!"GlobalDataStyle", !"Xe"}
!363 = !{!"NeedsBTD", i1 true}
!364 = !{!"EnableTextureIndirection", i1 false}
!365 = !{!"EnableSamplerIndirection", i1 false}
!366 = !{!"samplerStateStride", i32 0}
!367 = !{!"samplerStateOffset", i32 0}
!368 = !{!"textureStateStride", i32 0}
!369 = !{!"textureStateOffset", i32 0}
!370 = !{!"CurUniqueIndirectIdx", i32 0}
!371 = !{!"inlineDynTextures"}
!372 = !{!"inlineResInfoData"}
!373 = !{!"immConstant", !374, !375, !376}
!374 = !{!"data"}
!375 = !{!"sizes"}
!376 = !{!"zeroIdxs"}
!377 = !{!"stringConstants"}
!378 = !{!"inlineBuffers", !379, !383, !384}
!379 = !{!"inlineBuffersVec[0]", !380, !381, !382}
!380 = !{!"alignment", i32 0}
!381 = !{!"allocSize", i64 0}
!382 = !{!"Buffer"}
!383 = !{!"inlineBuffersVec[1]", !380, !381, !382}
!384 = !{!"inlineBuffersVec[2]", !380, !381, !382}
!385 = !{!"GlobalPointerProgramBinaryInfos"}
!386 = !{!"ConstantPointerProgramBinaryInfos"}
!387 = !{!"GlobalBufferAddressRelocInfo"}
!388 = !{!"ConstantBufferAddressRelocInfo"}
!389 = !{!"forceLscCacheList"}
!390 = !{!"SrvMap"}
!391 = !{!"RootConstantBufferOffsetInBytes"}
!392 = !{!"RasterizerOrderedByteAddressBuffer"}
!393 = !{!"RasterizerOrderedViews"}
!394 = !{!"MinNOSPushConstantSize", i32 0}
!395 = !{!"inlineProgramScopeOffsets"}
!396 = !{!"shaderData", !397}
!397 = !{!"numReplicas", i32 0}
!398 = !{!"URBInfo", !399, !400, !401}
!399 = !{!"has64BVertexHeaderInput", i1 false}
!400 = !{!"has64BVertexHeaderOutput", i1 false}
!401 = !{!"hasVertexHeader", i1 true}
!402 = !{!"m_ForcePullModel", i1 false}
!403 = !{!"UseBindlessImage", i1 false}
!404 = !{!"enableRangeReduce", i1 false}
!405 = !{!"disableNewTrigFuncRangeReduction", i1 false}
!406 = !{!"enableFRemToSRemOpt", i1 false}
!407 = !{!"enableSampleptrToLdmsptrSample0", i1 false}
!408 = !{!"enableSampleLptrToLdmsptrSample0", i1 false}
!409 = !{!"WaForceSIMD32MicropolyRasterize", i1 false}
!410 = !{!"allowMatchMadOptimizationforVS", i1 false}
!411 = !{!"disableMatchMadOptimizationForCS", i1 false}
!412 = !{!"disableMemOptforNegativeOffsetLoads", i1 false}
!413 = !{!"enableThreeWayLoadSpiltOpt", i1 false}
!414 = !{!"statefulResourcesNotAliased", i1 false}
!415 = !{!"disableMixMode", i1 false}
!416 = !{!"genericAccessesResolved", i1 false}
!417 = !{!"disableSeparateSpillPvtScratchSpace", i1 false}
!418 = !{!"disableSeparateScratchWA", i1 false}
!419 = !{!"PrivateMemoryPerFG"}
!420 = !{!"m_OptsToDisable"}
!421 = !{!"capabilities", !422}
!422 = !{!"globalVariableDecorationsINTEL", i1 false}
!423 = !{!"m_ShaderResourceViewMcsMask", !424, !425}
!424 = !{!"m_ShaderResourceViewMcsMaskVec[0]", i64 0}
!425 = !{!"m_ShaderResourceViewMcsMaskVec[1]", i64 0}
!426 = !{!"computedDepthMode", i32 0}
!427 = !{!"isHDCFastClearShader", i1 false}
!428 = !{!"argRegisterReservations", !429}
!429 = !{!"argRegisterReservationsVec[0]", i32 0}
!430 = !{!"SIMD16_SpillThreshold", i8 0}
!431 = !{!"SIMD32_SpillThreshold", i8 0}
!432 = !{!"m_CacheControlOption", !433, !434, !435, !436}
!433 = !{!"LscLoadCacheControlOverride", i8 0}
!434 = !{!"LscStoreCacheControlOverride", i8 0}
!435 = !{!"TgmLoadCacheControlOverride", i8 0}
!436 = !{!"TgmStoreCacheControlOverride", i8 0}
!437 = !{i32 2, i32 0}
!438 = !{!"clang version 14.0.5"}
!439 = !{i32 1, !"wchar_size", i32 4}
!440 = !{i32 1, i32 0, i32 0, i32 0, i32 1, i32 0, i32 0, i32 1, i32 0, i32 0}
!441 = !{!"none", !"none", !"none", !"none", !"none", !"none", !"none", !"none", !"none", !"none"}
!442 = !{!"float*", !"class.sycl::_V1::range", !"class.sycl::_V1::range", !"long", !"class.sycl::_V1::ext::oneapi::bfloat16*", !"class.sycl::_V1::range", !"class.sycl::_V1::range", !"class.sycl::_V1::ext::oneapi::bfloat16*", !"class.sycl::_V1::range", !"class.sycl::_V1::range"}
!443 = !{!"", !"", !"", !"", !"", !"", !"", !"", !"", !""}
!444 = !{!445, !447, !447, !450, !451, !447, !447, !451, !447, !447}
!445 = !{!446}
!446 = !{i32 44, i32 4}
!447 = !{!448, !449}
!448 = !{i32 38, i32 2}
!449 = !{i32 44, i32 8}
!450 = !{}
!451 = !{!452}
!452 = !{i32 44, i32 2}
!453 = !{!454}
!454 = !{i32 4469}
!455 = !{!456, !456, i64 0}
!456 = !{!"int", !457, i64 0}
!457 = !{!"omnipotent char", !458, i64 0}
!458 = !{!"Simple C/C++ TBAA"}
!459 = !{!454, !460}
!460 = !{i32 4470}
