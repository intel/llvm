; ------------------------------------------------
; OCL_asm40d739ce99ad8281_0029_Unify_after_FunctionIntegration_Inlining.ll
; LLVM major version: 14
; ------------------------------------------------

Printing <null> Function
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
  %48 = call spir_func i32 @__builtin_IB_get_group_id(i32 noundef 1) #6
  %49 = call spir_func i32 @__builtin_IB_get_enqueued_local_size(i32 noundef 1) #6
  %50 = mul i32 %49, %48
  %51 = call spir_func i32 @__builtin_IB_get_local_id_y() #6
  %52 = icmp ult i32 %51, 65536
  call void @llvm.assume(i1 %52) #3
  %53 = add i32 %51, %50
  %54 = call spir_func i32 @__builtin_IB_get_global_offset(i32 noundef 1) #6
  %55 = add i32 %53, %54
  %56 = zext i32 %55 to i64
  %57 = icmp ult i64 %56, 2147483648
  call void @llvm.assume(i1 %57)
  %58 = call spir_func i32 @__builtin_IB_get_group_id(i32 noundef 0) #6
  %59 = call spir_func i32 @__builtin_IB_get_enqueued_local_size(i32 noundef 0) #6
  %60 = mul i32 %59, %58
  %61 = call spir_func i32 @__builtin_IB_get_local_id_x() #6
  %62 = icmp ult i32 %61, 65536
  call void @llvm.assume(i1 %62) #3
  %63 = add i32 %61, %60
  %64 = call spir_func i32 @__builtin_IB_get_global_offset(i32 noundef 0) #6
  %65 = add i32 %63, %64
  %66 = zext i32 %65 to i64
  %67 = icmp ult i64 %66, 2147483648
  call void @llvm.assume(i1 %67)
  %68 = call spir_func i32 @__builtin_IB_get_local_id_y() #6
  %69 = icmp ult i32 %68, 65536
  call void @llvm.assume(i1 %69) #3
  %70 = zext i32 %68 to i64
  %71 = icmp ult i64 %70, 2147483648
  call void @llvm.assume(i1 %71)
  %72 = sub nsw i64 %56, %70, !spirv.Decorations !453
  %73 = call spir_func i32 @__builtin_IB_get_local_id_x() #6
  %74 = icmp ult i32 %73, 65536
  call void @llvm.assume(i1 %74) #3
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
  %87 = call spir_func i32 @__builtin_IB_get_local_size(i32 noundef 0) #6
  %88 = call spir_func i32 @__builtin_IB_get_local_size(i32 noundef 1) #6
  %89 = mul i32 %88, %87
  %90 = call spir_func i32 @__builtin_IB_get_local_size(i32 noundef 2) #6
  %91 = mul i32 %89, %90
  %92 = call spir_func i32 @__builtin_IB_get_simd_size() #7
  %93 = add i32 %92, -1
  %94 = and i32 %93, %91
  %95 = icmp eq i32 %94, 0
  br i1 %95, label %120, label %96

96:                                               ; preds = %10
  %97 = call spir_func i32 @__builtin_IB_get_local_id_z() #6
  %98 = call spir_func i32 @__builtin_IB_get_local_size(i32 noundef 1) #6
  %99 = mul i32 %98, %97
  %100 = call spir_func i32 @__builtin_IB_get_local_id_y() #6
  %101 = add i32 %99, %100
  %102 = call spir_func i32 @__builtin_IB_get_local_size(i32 noundef 0) #6
  %103 = mul i32 %101, %102
  %104 = call spir_func i32 @__builtin_IB_get_local_id_x() #6
  %105 = add i32 %103, %104
  %106 = call spir_func i32 @__builtin_IB_get_simd_size() #7
  %107 = udiv i32 %105, %106
  %108 = call spir_func i32 @__builtin_IB_get_local_size(i32 noundef 0) #6
  %109 = call spir_func i32 @__builtin_IB_get_local_size(i32 noundef 1) #6
  %110 = mul i32 %109, %108
  %111 = call spir_func i32 @__builtin_IB_get_local_size(i32 noundef 2) #6
  %112 = mul i32 %110, %111
  %113 = call spir_func i32 @__builtin_IB_get_simd_size() #7
  %114 = add i32 %112, -1
  %115 = add i32 %114, %113
  %116 = call spir_func i32 @__builtin_IB_get_simd_size() #7
  %117 = udiv i32 %115, %116
  %118 = add i32 %117, -1
  %119 = icmp ult i32 %107, %118
  br i1 %119, label %120, label %_Z18get_sub_group_sizev.exit.i4

120:                                              ; preds = %96, %10
  %121 = call spir_func i32 @__builtin_IB_get_simd_size() #7
  br label %_Z18get_sub_group_sizev.exit.i4

_Z18get_sub_group_sizev.exit.i4:                  ; preds = %120, %96
  %122 = phi i32 [ %121, %120 ], [ %94, %96 ]
  %123 = bitcast i8 addrspace(4)* %.ascast.i3 to i32 addrspace(4)*
  %124 = call spir_func i32 @__builtin_IB_get_simd_id() #7
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
  %173 = call spir_func i32 @__builtin_IB_get_local_size(i32 noundef 0) #6
  %174 = call spir_func i32 @__builtin_IB_get_local_size(i32 noundef 1) #6
  %175 = mul i32 %174, %173
  %176 = call spir_func i32 @__builtin_IB_get_local_size(i32 noundef 2) #6
  %177 = mul i32 %175, %176
  %178 = call spir_func i32 @__builtin_IB_get_simd_size() #7
  %179 = add i32 %178, -1
  %180 = and i32 %179, %177
  %181 = icmp eq i32 %180, 0
  br i1 %181, label %206, label %182

182:                                              ; preds = %168
  %183 = call spir_func i32 @__builtin_IB_get_local_id_z() #6
  %184 = call spir_func i32 @__builtin_IB_get_local_size(i32 noundef 1) #6
  %185 = mul i32 %184, %183
  %186 = call spir_func i32 @__builtin_IB_get_local_id_y() #6
  %187 = add i32 %185, %186
  %188 = call spir_func i32 @__builtin_IB_get_local_size(i32 noundef 0) #6
  %189 = mul i32 %187, %188
  %190 = call spir_func i32 @__builtin_IB_get_local_id_x() #6
  %191 = add i32 %189, %190
  %192 = call spir_func i32 @__builtin_IB_get_simd_size() #7
  %193 = udiv i32 %191, %192
  %194 = call spir_func i32 @__builtin_IB_get_local_size(i32 noundef 0) #6
  %195 = call spir_func i32 @__builtin_IB_get_local_size(i32 noundef 1) #6
  %196 = mul i32 %195, %194
  %197 = call spir_func i32 @__builtin_IB_get_local_size(i32 noundef 2) #6
  %198 = mul i32 %196, %197
  %199 = call spir_func i32 @__builtin_IB_get_simd_size() #7
  %200 = add i32 %198, -1
  %201 = add i32 %200, %199
  %202 = call spir_func i32 @__builtin_IB_get_simd_size() #7
  %203 = udiv i32 %201, %202
  %204 = add i32 %203, -1
  %205 = icmp ult i32 %193, %204
  br i1 %205, label %206, label %_Z18get_sub_group_sizev.exit.i6

206:                                              ; preds = %182, %168
  %207 = call spir_func i32 @__builtin_IB_get_simd_size() #7
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
  %217 = call spir_func i32 @__builtin_IB_simd_block_read_1_global(i32 addrspace(1)* noundef %216) #7
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
  %225 = call spir_func i32 @__builtin_IB_get_local_size(i32 noundef 0) #6
  %226 = call spir_func i32 @__builtin_IB_get_local_size(i32 noundef 1) #6
  %227 = mul i32 %226, %225
  %228 = call spir_func i32 @__builtin_IB_get_local_size(i32 noundef 2) #6
  %229 = mul i32 %227, %228
  %230 = call spir_func i32 @__builtin_IB_get_simd_size() #7
  %231 = add i32 %230, -1
  %232 = and i32 %231, %229
  %233 = icmp eq i32 %232, 0
  br i1 %233, label %258, label %234

234:                                              ; preds = %__builtin_spriv_OpJointMatrixLoadINTEL_PackedA_RowMajor_32x16_i16_32_global_v8i8_pi32_i32.24.exit
  %235 = call spir_func i32 @__builtin_IB_get_local_id_z() #6
  %236 = call spir_func i32 @__builtin_IB_get_local_size(i32 noundef 1) #6
  %237 = mul i32 %236, %235
  %238 = call spir_func i32 @__builtin_IB_get_local_id_y() #6
  %239 = add i32 %237, %238
  %240 = call spir_func i32 @__builtin_IB_get_local_size(i32 noundef 0) #6
  %241 = mul i32 %239, %240
  %242 = call spir_func i32 @__builtin_IB_get_local_id_x() #6
  %243 = add i32 %241, %242
  %244 = call spir_func i32 @__builtin_IB_get_simd_size() #7
  %245 = udiv i32 %243, %244
  %246 = call spir_func i32 @__builtin_IB_get_local_size(i32 noundef 0) #6
  %247 = call spir_func i32 @__builtin_IB_get_local_size(i32 noundef 1) #6
  %248 = mul i32 %247, %246
  %249 = call spir_func i32 @__builtin_IB_get_local_size(i32 noundef 2) #6
  %250 = mul i32 %248, %249
  %251 = call spir_func i32 @__builtin_IB_get_simd_size() #7
  %252 = add i32 %250, -1
  %253 = add i32 %252, %251
  %254 = call spir_func i32 @__builtin_IB_get_simd_size() #7
  %255 = udiv i32 %253, %254
  %256 = add i32 %255, -1
  %257 = icmp ult i32 %245, %256
  br i1 %257, label %258, label %_Z18get_sub_group_sizev.exit.i8

258:                                              ; preds = %234, %__builtin_spriv_OpJointMatrixLoadINTEL_PackedA_RowMajor_32x16_i16_32_global_v8i8_pi32_i32.24.exit
  %259 = call spir_func i32 @__builtin_IB_get_simd_size() #7
  br label %_Z18get_sub_group_sizev.exit.i8

_Z18get_sub_group_sizev.exit.i8:                  ; preds = %258, %234
  %260 = phi i32 [ %259, %258 ], [ %232, %234 ]
  %261 = bitcast i8 addrspace(4)* %.ascast.i7 to i32 addrspace(4)*
  %262 = call spir_func i32 @__builtin_IB_get_simd_id() #7
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
  %291 = call spir_func i32 @__builtin_IB_get_local_size(i32 noundef 0) #6
  %292 = call spir_func i32 @__builtin_IB_get_local_size(i32 noundef 1) #6
  %293 = mul i32 %292, %291
  %294 = call spir_func i32 @__builtin_IB_get_local_size(i32 noundef 2) #6
  %295 = mul i32 %293, %294
  %296 = call spir_func i32 @__builtin_IB_get_simd_size() #7
  %297 = add i32 %296, -1
  %298 = and i32 %297, %295
  %299 = icmp eq i32 %298, 0
  br i1 %299, label %324, label %300

300:                                              ; preds = %289
  %301 = call spir_func i32 @__builtin_IB_get_local_id_z() #6
  %302 = call spir_func i32 @__builtin_IB_get_local_size(i32 noundef 1) #6
  %303 = mul i32 %302, %301
  %304 = call spir_func i32 @__builtin_IB_get_local_id_y() #6
  %305 = add i32 %303, %304
  %306 = call spir_func i32 @__builtin_IB_get_local_size(i32 noundef 0) #6
  %307 = mul i32 %305, %306
  %308 = call spir_func i32 @__builtin_IB_get_local_id_x() #6
  %309 = add i32 %307, %308
  %310 = call spir_func i32 @__builtin_IB_get_simd_size() #7
  %311 = udiv i32 %309, %310
  %312 = call spir_func i32 @__builtin_IB_get_local_size(i32 noundef 0) #6
  %313 = call spir_func i32 @__builtin_IB_get_local_size(i32 noundef 1) #6
  %314 = mul i32 %313, %312
  %315 = call spir_func i32 @__builtin_IB_get_local_size(i32 noundef 2) #6
  %316 = mul i32 %314, %315
  %317 = call spir_func i32 @__builtin_IB_get_simd_size() #7
  %318 = add i32 %316, -1
  %319 = add i32 %318, %317
  %320 = call spir_func i32 @__builtin_IB_get_simd_size() #7
  %321 = udiv i32 %319, %320
  %322 = add i32 %321, -1
  %323 = icmp ult i32 %311, %322
  br i1 %323, label %324, label %_Z18get_sub_group_sizev.exit.i

324:                                              ; preds = %300, %289
  %325 = call spir_func i32 @__builtin_IB_get_simd_size() #7
  br label %_Z18get_sub_group_sizev.exit.i

_Z18get_sub_group_sizev.exit.i:                   ; preds = %324, %300
  %326 = phi i32 [ %325, %324 ], [ %298, %300 ]
  %327 = bitcast i8 addrspace(4)* %.ascast.i to i32 addrspace(4)*
  %328 = call spir_func i32 @__builtin_IB_get_simd_id() #7
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

Printing <null> Function
