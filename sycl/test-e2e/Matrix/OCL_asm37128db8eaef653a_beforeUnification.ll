; ------------------------------------------------
; OCL_asm37128db8eaef653a_beforeUnification.ll
; ------------------------------------------------
target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v16:16:16-v24:32:32-v32:32:32-v48:64:64-v64:64:64-v96:128:128-v128:128:128-v192:256:256-v256:256:256-v512:512:512-v1024:1024:1024"
target triple = "spir64-unknown-unknown"

%"class.sycl::_V1::ext::oneapi::bfloat16" = type { i16 }
%"class.sycl::_V1::range" = type { %"class.sycl::_V1::detail::array" }
%"class.sycl::_V1::detail::array" = type { [2 x i64] }
%"struct.sycl::_V1::ext::oneapi::experimental::matrix::joint_matrix.10" = type { %spirv.JointMatrixINTEL._short_8_16_0_3_0 addrspace(1)* }
%spirv.JointMatrixINTEL._short_8_16_0_3_0 = type opaque
%"struct.sycl::_V1::ext::oneapi::experimental::matrix::joint_matrix.20" = type { %spirv.JointMatrixINTEL._float_8_8_3_3_2 addrspace(1)* }
%spirv.JointMatrixINTEL._float_8_8_3_3_2 = type opaque
%spirv.JointMatrixINTEL._short_16_8_2_3_1 = type opaque

; Function Attrs: nounwind
define spir_kernel void @_ZTS4multIN4sycl3_V13ext6oneapi8bfloat16ELm8ELm8ELm16EE(%"class.sycl::_V1::ext::oneapi::bfloat16" addrspace(1)* align 2 %0, %"class.sycl::_V1::range"* byval(%"class.sycl::_V1::range") align 8 %1, %"class.sycl::_V1::range"* byval(%"class.sycl::_V1::range") align 8 %2, i64 %3, %"class.sycl::_V1::ext::oneapi::bfloat16" addrspace(1)* align 2 %4, %"class.sycl::_V1::range"* byval(%"class.sycl::_V1::range") align 8 %5, %"class.sycl::_V1::range"* byval(%"class.sycl::_V1::range") align 8 %6, float addrspace(1)* align 4 %7, %"class.sycl::_V1::range"* byval(%"class.sycl::_V1::range") align 8 %8, %"class.sycl::_V1::range"* byval(%"class.sycl::_V1::range") align 8 %9) #0 !kernel_arg_addr_space !236 !kernel_arg_access_qual !237 !kernel_arg_type !238 !kernel_arg_type_qual !239 !kernel_arg_base_type !238 !kernel_arg_name !239 !spirv.ParameterDecorations !240 {
  %11 = alloca [2 x %"struct.sycl::_V1::ext::oneapi::experimental::matrix::joint_matrix.10"], align 8, !spirv.Decorations !248
  %12 = alloca [2 x %"struct.sycl::_V1::ext::oneapi::experimental::matrix::joint_matrix.20"], align 8, !spirv.Decorations !248
  call spir_func void @__itt_offload_wi_start_wrapper() #1
  %13 = bitcast %"class.sycl::_V1::range"* %1 to i64*
  %14 = getelementptr inbounds i64, i64* %13, i64 1
  %15 = load i64, i64* %14, align 8
  %16 = bitcast %"class.sycl::_V1::range"* %2 to i64*
  %17 = load i64, i64* %16, align 8
  %18 = bitcast %"class.sycl::_V1::range"* %2 to i64*
  %19 = getelementptr inbounds i64, i64* %18, i64 1
  %20 = load i64, i64* %19, align 8
  %21 = mul i64 %17, %15
  %22 = getelementptr %"class.sycl::_V1::ext::oneapi::bfloat16", %"class.sycl::_V1::ext::oneapi::bfloat16" addrspace(1)* %0, i64 %21
  %23 = getelementptr %"class.sycl::_V1::ext::oneapi::bfloat16", %"class.sycl::_V1::ext::oneapi::bfloat16" addrspace(1)* %22, i64 %20
  %24 = bitcast %"class.sycl::_V1::range"* %5 to i64*
  %25 = getelementptr inbounds i64, i64* %24, i64 1
  %26 = load i64, i64* %25, align 8
  %27 = bitcast %"class.sycl::_V1::range"* %6 to i64*
  %28 = load i64, i64* %27, align 8
  %29 = bitcast %"class.sycl::_V1::range"* %6 to i64*
  %30 = getelementptr inbounds i64, i64* %29, i64 1
  %31 = load i64, i64* %30, align 8
  %32 = mul i64 %28, %26
  %33 = getelementptr %"class.sycl::_V1::ext::oneapi::bfloat16", %"class.sycl::_V1::ext::oneapi::bfloat16" addrspace(1)* %4, i64 %32
  %34 = getelementptr %"class.sycl::_V1::ext::oneapi::bfloat16", %"class.sycl::_V1::ext::oneapi::bfloat16" addrspace(1)* %33, i64 %31
  %35 = bitcast %"class.sycl::_V1::range"* %8 to i64*
  %36 = getelementptr inbounds i64, i64* %35, i64 1
  %37 = load i64, i64* %36, align 8
  %38 = bitcast %"class.sycl::_V1::range"* %9 to i64*
  %39 = load i64, i64* %38, align 8
  %40 = bitcast %"class.sycl::_V1::range"* %9 to i64*
  %41 = getelementptr inbounds i64, i64* %40, i64 1
  %42 = load i64, i64* %41, align 8
  %43 = mul i64 %39, %37
  %44 = getelementptr float, float addrspace(1)* %7, i64 %43
  %45 = getelementptr float, float addrspace(1)* %44, i64 %42
  %46 = call spir_func i64 @_Z33__spirv_BuiltInGlobalInvocationIdi(i32 1) #5
  %47 = icmp ult i64 %46, 2147483648
  call void @llvm.assume(i1 %47)
  %48 = call spir_func i64 @_Z33__spirv_BuiltInGlobalInvocationIdi(i32 0) #5
  %49 = icmp ult i64 %48, 2147483648
  call void @llvm.assume(i1 %49)
  %50 = call spir_func i64 @_Z32__spirv_BuiltInLocalInvocationIdi(i32 1) #5
  %51 = icmp ult i64 %50, 2147483648
  call void @llvm.assume(i1 %51)
  %52 = sub nsw i64 %46, %50, !spirv.Decorations !249
  %53 = call spir_func i64 @_Z32__spirv_BuiltInLocalInvocationIdi(i32 0) #5
  %54 = icmp ult i64 %53, 2147483648
  call void @llvm.assume(i1 %54)
  %55 = sub nsw i64 %48, %53, !spirv.Decorations !249
  %56 = bitcast [2 x %"struct.sycl::_V1::ext::oneapi::experimental::matrix::joint_matrix.10"]* %11 to i8*
  call void @llvm.lifetime.start.p0i8(i64 16, i8* %56)
  %57 = getelementptr inbounds [2 x %"struct.sycl::_V1::ext::oneapi::experimental::matrix::joint_matrix.10"], [2 x %"struct.sycl::_V1::ext::oneapi::experimental::matrix::joint_matrix.10"]* %11, i64 0, i64 0
  %58 = getelementptr inbounds [2 x %"struct.sycl::_V1::ext::oneapi::experimental::matrix::joint_matrix.10"], [2 x %"struct.sycl::_V1::ext::oneapi::experimental::matrix::joint_matrix.10"]* %11, i64 1
  br label %59

59:                                               ; preds = %59, %10
  %60 = phi %"struct.sycl::_V1::ext::oneapi::experimental::matrix::joint_matrix.10"* [ %57, %10 ], [ %61, %59 ]
  %61 = getelementptr inbounds %"struct.sycl::_V1::ext::oneapi::experimental::matrix::joint_matrix.10", %"struct.sycl::_V1::ext::oneapi::experimental::matrix::joint_matrix.10"* %60, i64 1
  %62 = bitcast [2 x %"struct.sycl::_V1::ext::oneapi::experimental::matrix::joint_matrix.10"]* %58 to %"struct.sycl::_V1::ext::oneapi::experimental::matrix::joint_matrix.10"*
  %63 = ptrtoint %"struct.sycl::_V1::ext::oneapi::experimental::matrix::joint_matrix.10"* %61 to i64
  %64 = ptrtoint %"struct.sycl::_V1::ext::oneapi::experimental::matrix::joint_matrix.10"* %62 to i64
  %65 = icmp eq i64 %63, %64
  br i1 %65, label %66, label %59

66:                                               ; preds = %59
  %67 = bitcast [2 x %"struct.sycl::_V1::ext::oneapi::experimental::matrix::joint_matrix.20"]* %12 to i8*
  call void @llvm.lifetime.start.p0i8(i64 16, i8* %67)
  %68 = getelementptr inbounds [2 x %"struct.sycl::_V1::ext::oneapi::experimental::matrix::joint_matrix.20"], [2 x %"struct.sycl::_V1::ext::oneapi::experimental::matrix::joint_matrix.20"]* %12, i64 0, i64 0
  %69 = getelementptr inbounds [2 x %"struct.sycl::_V1::ext::oneapi::experimental::matrix::joint_matrix.20"], [2 x %"struct.sycl::_V1::ext::oneapi::experimental::matrix::joint_matrix.20"]* %12, i64 1
  br label %70

70:                                               ; preds = %70, %66
  %71 = phi %"struct.sycl::_V1::ext::oneapi::experimental::matrix::joint_matrix.20"* [ %68, %66 ], [ %72, %70 ]
  %72 = getelementptr inbounds %"struct.sycl::_V1::ext::oneapi::experimental::matrix::joint_matrix.20", %"struct.sycl::_V1::ext::oneapi::experimental::matrix::joint_matrix.20"* %71, i64 1
  %73 = bitcast [2 x %"struct.sycl::_V1::ext::oneapi::experimental::matrix::joint_matrix.20"]* %69 to %"struct.sycl::_V1::ext::oneapi::experimental::matrix::joint_matrix.20"*
  %74 = ptrtoint %"struct.sycl::_V1::ext::oneapi::experimental::matrix::joint_matrix.20"* %72 to i64
  %75 = ptrtoint %"struct.sycl::_V1::ext::oneapi::experimental::matrix::joint_matrix.20"* %73 to i64
  %76 = icmp eq i64 %74, %75
  br i1 %76, label %.preheader, label %70

.preheader:                                       ; preds = %70
  br label %77

77:                                               ; preds = %91, %.preheader
  %78 = phi i32 [ %96, %91 ], [ 0, %.preheader ]
  %79 = icmp ult i32 %78, 2
  br i1 %79, label %91, label %80

80:                                               ; preds = %77
  %81 = add i64 %21, %20
  %82 = sub i64 0, %81
  %83 = getelementptr inbounds %"class.sycl::_V1::ext::oneapi::bfloat16", %"class.sycl::_V1::ext::oneapi::bfloat16" addrspace(1)* %23, i64 %82
  %84 = udiv i64 %55, %3
  %85 = shl i64 %84, 4
  %86 = getelementptr %"class.sycl::_V1::ext::oneapi::bfloat16", %"class.sycl::_V1::ext::oneapi::bfloat16" addrspace(1)* %83, i64 %85
  %87 = add i64 %32, %31
  %88 = sub i64 0, %87
  %89 = getelementptr inbounds %"class.sycl::_V1::ext::oneapi::bfloat16", %"class.sycl::_V1::ext::oneapi::bfloat16" addrspace(1)* %34, i64 %88
  %90 = shl nsw i64 %52, 9, !spirv.Decorations !249
  br label %97

91:                                               ; preds = %77
  %92 = zext i32 %78 to i64
  %93 = call spir_func %spirv.JointMatrixINTEL._float_8_8_3_3_2 addrspace(1)* @_Z26__spirv_CompositeConstructf(float 1.000000e+00) #0
  %94 = getelementptr inbounds [2 x %"struct.sycl::_V1::ext::oneapi::experimental::matrix::joint_matrix.20"], [2 x %"struct.sycl::_V1::ext::oneapi::experimental::matrix::joint_matrix.20"]* %12, i64 0, i64 %92
  %95 = bitcast %"struct.sycl::_V1::ext::oneapi::experimental::matrix::joint_matrix.20"* %94 to %spirv.JointMatrixINTEL._float_8_8_3_3_2 addrspace(1)**
  store %spirv.JointMatrixINTEL._float_8_8_3_3_2 addrspace(1)* %93, %spirv.JointMatrixINTEL._float_8_8_3_3_2 addrspace(1)** %95, align 8
  %96 = add nuw nsw i32 %78, 1, !spirv.Decorations !251
  br label %77

97:                                               ; preds = %118, %80
  %98 = phi i32 [ %119, %118 ], [ 0, %80 ]
  %99 = icmp ult i32 %98, 2
  br i1 %99, label %108, label %100

100:                                              ; preds = %97
  %101 = add i64 %43, %42
  %102 = sub i64 0, %101
  %103 = getelementptr inbounds float, float addrspace(1)* %45, i64 %102
  %104 = shl nsw i64 %52, 8, !spirv.Decorations !249
  %105 = udiv i64 %55, %3
  %106 = shl i64 %105, 3
  %107 = getelementptr float, float addrspace(1)* %103, i64 %106
  br label %134

108:                                              ; preds = %97
  %109 = shl nuw nsw i32 %98, 4, !spirv.Decorations !251
  %110 = zext i32 %109 to i64
  %111 = shl nuw nsw i64 %110, 4, !spirv.Decorations !251
  %112 = getelementptr %"class.sycl::_V1::ext::oneapi::bfloat16", %"class.sycl::_V1::ext::oneapi::bfloat16" addrspace(1)* %86, i64 %111
  %113 = call spir_func %spirv.JointMatrixINTEL._short_16_8_2_3_1 addrspace(1)* @"_Z80__spirv_JointMatrixLoadINTEL_RPU3AS142__spirv_JointMatrixINTEL__short_16_8_2_3_1PU3AS138class.sycl::_V1::ext::oneapi::bfloat16liii"(%"class.sycl::_V1::ext::oneapi::bfloat16" addrspace(1)* %112, i64 32, i32 2, i32 3, i32 0) #0
  %114 = getelementptr %"class.sycl::_V1::ext::oneapi::bfloat16", %"class.sycl::_V1::ext::oneapi::bfloat16" addrspace(1)* %89, i64 %110
  br label %115

115:                                              ; preds = %120, %108
  %116 = phi i32 [ 0, %108 ], [ %133, %120 ]
  %117 = icmp ult i32 %116, 2
  br i1 %117, label %120, label %118

118:                                              ; preds = %115
  %119 = add nuw nsw i32 %98, 1, !spirv.Decorations !251
  br label %97

120:                                              ; preds = %115
  %121 = zext i32 %116 to i64
  %122 = shl nuw nsw i64 %121, 8, !spirv.Decorations !251
  %123 = or i64 %90, %122
  %124 = getelementptr %"class.sycl::_V1::ext::oneapi::bfloat16", %"class.sycl::_V1::ext::oneapi::bfloat16" addrspace(1)* %114, i64 %123
  %125 = call spir_func %spirv.JointMatrixINTEL._short_8_16_0_3_0 addrspace(1)* @"_Z80__spirv_JointMatrixLoadINTEL_RPU3AS142__spirv_JointMatrixINTEL__short_8_16_0_3_0PU3AS138class.sycl::_V1::ext::oneapi::bfloat16liii"(%"class.sycl::_V1::ext::oneapi::bfloat16" addrspace(1)* %124, i64 32, i32 0, i32 3, i32 0) #0
  %126 = getelementptr inbounds [2 x %"struct.sycl::_V1::ext::oneapi::experimental::matrix::joint_matrix.10"], [2 x %"struct.sycl::_V1::ext::oneapi::experimental::matrix::joint_matrix.10"]* %11, i64 0, i64 %121
  %127 = bitcast %"struct.sycl::_V1::ext::oneapi::experimental::matrix::joint_matrix.10"* %126 to %spirv.JointMatrixINTEL._short_8_16_0_3_0 addrspace(1)**
  store %spirv.JointMatrixINTEL._short_8_16_0_3_0 addrspace(1)* %125, %spirv.JointMatrixINTEL._short_8_16_0_3_0 addrspace(1)** %127, align 8
  %128 = getelementptr inbounds [2 x %"struct.sycl::_V1::ext::oneapi::experimental::matrix::joint_matrix.20"], [2 x %"struct.sycl::_V1::ext::oneapi::experimental::matrix::joint_matrix.20"]* %12, i64 0, i64 %121
  %129 = bitcast %"struct.sycl::_V1::ext::oneapi::experimental::matrix::joint_matrix.20"* %128 to %spirv.JointMatrixINTEL._float_8_8_3_3_2 addrspace(1)**
  %130 = load %spirv.JointMatrixINTEL._float_8_8_3_3_2 addrspace(1)*, %spirv.JointMatrixINTEL._float_8_8_3_3_2 addrspace(1)** %129, align 8
  %131 = call spir_func %spirv.JointMatrixINTEL._float_8_8_3_3_2 addrspace(1)* @_Z27__spirv_JointMatrixMadINTELPU3AS142__spirv_JointMatrixINTEL__short_8_16_0_3_0PU3AS142__spirv_JointMatrixINTEL__short_16_8_2_3_1PU3AS141__spirv_JointMatrixINTEL__float_8_8_3_3_2i(%spirv.JointMatrixINTEL._short_8_16_0_3_0 addrspace(1)* %125, %spirv.JointMatrixINTEL._short_16_8_2_3_1 addrspace(1)* %113, %spirv.JointMatrixINTEL._float_8_8_3_3_2 addrspace(1)* %130, i32 3) #0
  %132 = bitcast %"struct.sycl::_V1::ext::oneapi::experimental::matrix::joint_matrix.20"* %128 to %spirv.JointMatrixINTEL._float_8_8_3_3_2 addrspace(1)**
  store %spirv.JointMatrixINTEL._float_8_8_3_3_2 addrspace(1)* %131, %spirv.JointMatrixINTEL._float_8_8_3_3_2 addrspace(1)** %132, align 8
  %133 = add nuw nsw i32 %116, 1, !spirv.Decorations !251
  br label %115

134:                                              ; preds = %137, %100
  %135 = phi i32 [ %145, %137 ], [ 0, %100 ]
  %136 = icmp ult i32 %135, 2
  br i1 %136, label %137, label %146

137:                                              ; preds = %134
  %138 = zext i32 %135 to i64
  %139 = shl nuw nsw i64 %138, 7, !spirv.Decorations !251
  %140 = or i64 %104, %139
  %141 = getelementptr float, float addrspace(1)* %107, i64 %140
  %142 = getelementptr inbounds [2 x %"struct.sycl::_V1::ext::oneapi::experimental::matrix::joint_matrix.20"], [2 x %"struct.sycl::_V1::ext::oneapi::experimental::matrix::joint_matrix.20"]* %12, i64 0, i64 %138
  %143 = bitcast %"struct.sycl::_V1::ext::oneapi::experimental::matrix::joint_matrix.20"* %142 to %spirv.JointMatrixINTEL._float_8_8_3_3_2 addrspace(1)**
  %144 = load %spirv.JointMatrixINTEL._float_8_8_3_3_2 addrspace(1)*, %spirv.JointMatrixINTEL._float_8_8_3_3_2 addrspace(1)** %143, align 8
  call spir_func void @_Z29__spirv_JointMatrixStoreINTELPU3AS1fPU3AS141__spirv_JointMatrixINTEL__float_8_8_3_3_2liii(float addrspace(1)* %141, %spirv.JointMatrixINTEL._float_8_8_3_3_2 addrspace(1)* %144, i64 16, i32 0, i32 3, i32 0) #0
  %145 = add nuw nsw i32 %135, 1, !spirv.Decorations !251
  br label %134

146:                                              ; preds = %134
  %147 = bitcast [2 x %"struct.sycl::_V1::ext::oneapi::experimental::matrix::joint_matrix.20"]* %12 to i8*
  call void @llvm.lifetime.end.p0i8(i64 16, i8* %147)
  %148 = bitcast [2 x %"struct.sycl::_V1::ext::oneapi::experimental::matrix::joint_matrix.10"]* %11 to i8*
  call void @llvm.lifetime.end.p0i8(i64 16, i8* %148)
  call spir_func void @__itt_offload_wi_finish_wrapper() #1
  ret void
}

; Function Attrs: alwaysinline nounwind
define spir_func void @__itt_offload_wi_start_wrapper() #1 {
  %1 = alloca [3 x i64], align 8, !spirv.Decorations !248
  br i1 true, label %25, label %2

2:                                                ; preds = %0
  %3 = bitcast [3 x i64]* %1 to i8*
  call void @llvm.lifetime.start.p0i8(i64 24, i8* %3)
  %4 = getelementptr inbounds [3 x i64], [3 x i64]* %1, i64 0, i64 0
  %5 = addrspacecast i64* %4 to i64 addrspace(4)*
  %6 = call spir_func i64 @_Z26__spirv_BuiltInWorkgroupIdi(i32 0) #5
  %7 = insertelement <3 x i64> undef, i64 %6, i32 0
  %8 = call spir_func i64 @_Z26__spirv_BuiltInWorkgroupIdi(i32 1) #5
  %9 = insertelement <3 x i64> %7, i64 %8, i32 1
  %10 = call spir_func i64 @_Z26__spirv_BuiltInWorkgroupIdi(i32 2) #5
  %11 = insertelement <3 x i64> %9, i64 %10, i32 2
  %12 = extractelement <3 x i64> %11, i32 0
  store i64 %12, i64* %4, align 8
  %13 = getelementptr inbounds [3 x i64], [3 x i64]* %1, i64 0, i64 1
  %14 = extractelement <3 x i64> %11, i32 1
  store i64 %14, i64* %13, align 8
  %15 = getelementptr inbounds [3 x i64], [3 x i64]* %1, i64 0, i64 2
  %16 = extractelement <3 x i64> %11, i32 2
  store i64 %16, i64* %15, align 8
  %17 = call spir_func i64 @_Z29__spirv_BuiltInGlobalLinearIdv() #5
  %18 = call spir_func i64 @_Z28__spirv_BuiltInWorkgroupSizei(i32 0) #5
  %19 = call spir_func i64 @_Z28__spirv_BuiltInWorkgroupSizei(i32 1) #5
  %20 = mul i64 %18, %19
  %21 = call spir_func i64 @_Z28__spirv_BuiltInWorkgroupSizei(i32 2) #5
  %22 = mul i64 %20, %21
  %23 = trunc i64 %22 to i32
  call spir_func void @__itt_offload_wi_start_stub(i64 addrspace(4)* %5, i64 %17, i32 %23) #3
  %24 = bitcast [3 x i64]* %1 to i8*
  call void @llvm.lifetime.end.p0i8(i64 24, i8* %24)
  br label %25

25:                                               ; preds = %2, %0
  ret void
}

; Function Attrs: argmemonly nofree nosync nounwind willreturn
declare void @llvm.lifetime.start.p0i8(i64 immarg, i8* nocapture) #2

; Function Attrs: noinline nounwind optnone
define spir_func void @__itt_offload_wi_start_stub(i64 addrspace(4)* %0, i64 %1, i32 %2) #3 {
  %4 = alloca i64 addrspace(4)*, align 8, !spirv.Decorations !248
  %5 = alloca i64, align 8, !spirv.Decorations !248
  %6 = alloca i32, align 4, !spirv.Decorations !246
  %7 = addrspacecast i64 addrspace(4)** %4 to i64 addrspace(4)* addrspace(4)*
  %8 = addrspacecast i64* %5 to i64 addrspace(4)*
  %9 = addrspacecast i32* %6 to i32 addrspace(4)*
  store i64 addrspace(4)* %0, i64 addrspace(4)* addrspace(4)* %7, align 8
  store i64 %1, i64 addrspace(4)* %8, align 8
  store i32 %2, i32 addrspace(4)* %9, align 4
  ret void
}

; Function Attrs: argmemonly nofree nosync nounwind willreturn
declare void @llvm.lifetime.end.p0i8(i64 immarg, i8* nocapture) #2

; Function Attrs: inaccessiblememonly nofree nosync nounwind willreturn
declare void @llvm.assume(i1 noundef) #4

; Function Attrs: nounwind
declare spir_func %spirv.JointMatrixINTEL._float_8_8_3_3_2 addrspace(1)* @_Z26__spirv_CompositeConstructf(float) #0

; Function Attrs: nounwind
declare spir_func %spirv.JointMatrixINTEL._short_16_8_2_3_1 addrspace(1)* @"_Z80__spirv_JointMatrixLoadINTEL_RPU3AS142__spirv_JointMatrixINTEL__short_16_8_2_3_1PU3AS138class.sycl::_V1::ext::oneapi::bfloat16liii"(%"class.sycl::_V1::ext::oneapi::bfloat16" addrspace(1)*, i64, i32, i32, i32) #0

; Function Attrs: nounwind
declare spir_func %spirv.JointMatrixINTEL._short_8_16_0_3_0 addrspace(1)* @"_Z80__spirv_JointMatrixLoadINTEL_RPU3AS142__spirv_JointMatrixINTEL__short_8_16_0_3_0PU3AS138class.sycl::_V1::ext::oneapi::bfloat16liii"(%"class.sycl::_V1::ext::oneapi::bfloat16" addrspace(1)*, i64, i32, i32, i32) #0

; Function Attrs: nounwind
declare spir_func %spirv.JointMatrixINTEL._float_8_8_3_3_2 addrspace(1)* @_Z27__spirv_JointMatrixMadINTELPU3AS142__spirv_JointMatrixINTEL__short_8_16_0_3_0PU3AS142__spirv_JointMatrixINTEL__short_16_8_2_3_1PU3AS141__spirv_JointMatrixINTEL__float_8_8_3_3_2i(%spirv.JointMatrixINTEL._short_8_16_0_3_0 addrspace(1)*, %spirv.JointMatrixINTEL._short_16_8_2_3_1 addrspace(1)*, %spirv.JointMatrixINTEL._float_8_8_3_3_2 addrspace(1)*, i32) #0

; Function Attrs: nounwind
declare spir_func void @_Z29__spirv_JointMatrixStoreINTELPU3AS1fPU3AS141__spirv_JointMatrixINTEL__float_8_8_3_3_2liii(float addrspace(1)*, %spirv.JointMatrixINTEL._float_8_8_3_3_2 addrspace(1)*, i64, i32, i32, i32) #0

; Function Attrs: alwaysinline nounwind
define spir_func void @__itt_offload_wi_finish_wrapper() #1 {
  %1 = alloca [3 x i64], align 8, !spirv.Decorations !248
  br i1 true, label %19, label %2

2:                                                ; preds = %0
  %3 = bitcast [3 x i64]* %1 to i8*
  call void @llvm.lifetime.start.p0i8(i64 24, i8* %3)
  %4 = getelementptr inbounds [3 x i64], [3 x i64]* %1, i64 0, i64 0
  %5 = addrspacecast i64* %4 to i64 addrspace(4)*
  %6 = call spir_func i64 @_Z26__spirv_BuiltInWorkgroupIdi(i32 0) #5
  %7 = insertelement <3 x i64> undef, i64 %6, i32 0
  %8 = call spir_func i64 @_Z26__spirv_BuiltInWorkgroupIdi(i32 1) #5
  %9 = insertelement <3 x i64> %7, i64 %8, i32 1
  %10 = call spir_func i64 @_Z26__spirv_BuiltInWorkgroupIdi(i32 2) #5
  %11 = insertelement <3 x i64> %9, i64 %10, i32 2
  %12 = extractelement <3 x i64> %11, i32 0
  store i64 %12, i64* %4, align 8
  %13 = getelementptr inbounds [3 x i64], [3 x i64]* %1, i64 0, i64 1
  %14 = extractelement <3 x i64> %11, i32 1
  store i64 %14, i64* %13, align 8
  %15 = getelementptr inbounds [3 x i64], [3 x i64]* %1, i64 0, i64 2
  %16 = extractelement <3 x i64> %11, i32 2
  store i64 %16, i64* %15, align 8
  %17 = call spir_func i64 @_Z29__spirv_BuiltInGlobalLinearIdv() #5
  call spir_func void @__itt_offload_wi_finish_stub(i64 addrspace(4)* %5, i64 %17) #3
  %18 = bitcast [3 x i64]* %1 to i8*
  call void @llvm.lifetime.end.p0i8(i64 24, i8* %18)
  br label %19

19:                                               ; preds = %2, %0
  ret void
}

; Function Attrs: noinline nounwind optnone
define spir_func void @__itt_offload_wi_finish_stub(i64 addrspace(4)* %0, i64 %1) #3 {
  %3 = alloca i64 addrspace(4)*, align 8, !spirv.Decorations !248
  %4 = alloca i64, align 8, !spirv.Decorations !248
  %5 = addrspacecast i64 addrspace(4)** %3 to i64 addrspace(4)* addrspace(4)*
  %6 = addrspacecast i64* %4 to i64 addrspace(4)*
  store i64 addrspace(4)* %0, i64 addrspace(4)* addrspace(4)* %5, align 8
  store i64 %1, i64 addrspace(4)* %6, align 8
  ret void
}

; Function Attrs: nounwind readnone willreturn
declare spir_func i64 @_Z33__spirv_BuiltInGlobalInvocationIdi(i32) #5

; Function Attrs: nounwind readnone willreturn
declare spir_func i64 @_Z32__spirv_BuiltInLocalInvocationIdi(i32) #5

; Function Attrs: nounwind readnone willreturn
declare spir_func i64 @_Z26__spirv_BuiltInWorkgroupIdi(i32) #5

; Function Attrs: nounwind readnone willreturn
declare spir_func i64 @_Z29__spirv_BuiltInGlobalLinearIdv() #5

; Function Attrs: nounwind readnone willreturn
declare spir_func i64 @_Z28__spirv_BuiltInWorkgroupSizei(i32) #5

attributes #0 = { nounwind }
attributes #1 = { alwaysinline nounwind }
attributes #2 = { argmemonly nofree nosync nounwind willreturn }
attributes #3 = { noinline nounwind optnone }
attributes #4 = { inaccessiblememonly nofree nosync nounwind willreturn }
attributes #5 = { nounwind readnone willreturn }

!spirv.MemoryModel = !{!0}
!opencl.enable.FP_CONTRACT = !{}
!spirv.Source = !{!1}
!opencl.spir.version = !{!2}
!opencl.ocl.version = !{!3}
!opencl.used.extensions = !{!4}
!opencl.used.optional.core.features = !{!4}
!spirv.Generator = !{!5}
!opencl.compiler.options = !{!6}
!igc.functions = !{}
!IGCMetadata = !{!7}

!0 = !{i32 2, i32 2}
!1 = !{i32 4, i32 100000}
!2 = !{i32 1, i32 2}
!3 = !{i32 1, i32 0}
!4 = !{}
!5 = !{i16 6, i16 14}
!6 = !{!"-ze-opt-level=O2"}
!7 = !{!"ModuleMD", !8, !9, !86, !87, !118, !134, !155, !165, !167, !168, !182, !183, !184, !185, !189, !190, !197, !198, !199, !200, !201, !202, !203, !204, !205, !206, !207, !209, !213, !214, !215, !216, !217, !218, !219, !220, !221, !222, !223, !224, !225, !226, !227, !229, !232, !233, !234}
!8 = !{!"isPrecise", i1 false}
!9 = !{!"compOpt", !10, !11, !12, !13, !14, !15, !16, !17, !18, !19, !20, !21, !22, !23, !24, !25, !26, !27, !28, !29, !30, !31, !32, !33, !34, !35, !36, !37, !38, !39, !40, !41, !42, !43, !44, !45, !46, !47, !48, !49, !50, !51, !52, !53, !54, !55, !56, !57, !58, !59, !60, !61, !62, !63, !64, !65, !66, !67, !68, !69, !70, !71, !72, !73, !74, !75, !76, !77, !78, !79, !80, !81, !82, !83, !84, !85}
!10 = !{!"DenormsAreZero", i1 false}
!11 = !{!"BFTFDenormsAreZero", i1 false}
!12 = !{!"CorrectlyRoundedDivSqrt", i1 false}
!13 = !{!"OptDisable", i1 false}
!14 = !{!"MadEnable", i1 false}
!15 = !{!"NoSignedZeros", i1 false}
!16 = !{!"NoNaNs", i1 false}
!17 = !{!"FloatRoundingMode", i32 0}
!18 = !{!"FloatCvtIntRoundingMode", i32 3}
!19 = !{!"LoadCacheDefault", i32 -1}
!20 = !{!"StoreCacheDefault", i32 -1}
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
!35 = !{!"GreaterThan4GBBufferRequired", i1 true}
!36 = !{!"DisableA64WA", i1 false}
!37 = !{!"ForceEnableA64WA", i1 false}
!38 = !{!"PushConstantsEnable", i1 true}
!39 = !{!"HasPositivePointerOffset", i1 false}
!40 = !{!"HasBufferOffsetArg", i1 false}
!41 = !{!"BufferOffsetArgOptional", i1 true}
!42 = !{!"replaceGlobalOffsetsByZero", i1 false}
!43 = !{!"forcePixelShaderSIMDMode", i32 0}
!44 = !{!"pixelShaderDoNotAbortOnSpill", i1 false}
!45 = !{!"UniformWGS", i1 false}
!46 = !{!"disableVertexComponentPacking", i1 false}
!47 = !{!"disablePartialVertexComponentPacking", i1 false}
!48 = !{!"PreferBindlessImages", i1 false}
!49 = !{!"UseBindlessMode", i1 false}
!50 = !{!"UseLegacyBindlessMode", i1 true}
!51 = !{!"disableMathRefactoring", i1 false}
!52 = !{!"atomicBranch", i1 false}
!53 = !{!"spillCompression", i1 false}
!54 = !{!"ForceInt32DivRemEmu", i1 false}
!55 = !{!"ForceInt32DivRemEmuSP", i1 false}
!56 = !{!"DisableFastestSingleCSSIMD", i1 false}
!57 = !{!"DisableFastestLinearScan", i1 false}
!58 = !{!"UseStatelessforPrivateMemory", i1 false}
!59 = !{!"EnableTakeGlobalAddress", i1 false}
!60 = !{!"IsLibraryCompilation", i1 false}
!61 = !{!"LibraryCompileSIMDSize", i32 0}
!62 = !{!"FastVISACompile", i1 false}
!63 = !{!"MatchSinCosPi", i1 false}
!64 = !{!"ExcludeIRFromZEBinary", i1 false}
!65 = !{!"EmitZeBinVISASections", i1 false}
!66 = !{!"FP64GenEmulationEnabled", i1 false}
!67 = !{!"FP64GenConvEmulationEnabled", i1 false}
!68 = !{!"allowDisableRematforCS", i1 false}
!69 = !{!"DisableIncSpillCostAllAddrTaken", i1 false}
!70 = !{!"DisableCPSOmaskWA", i1 false}
!71 = !{!"DisableFastestGopt", i1 false}
!72 = !{!"WaForceHalfPromotionComputeShader", i1 false}
!73 = !{!"WaForceHalfPromotionPixelVertexShader", i1 false}
!74 = !{!"DisableConstantCoalescing", i1 false}
!75 = !{!"EnableUndefAlphaOutputAsRed", i1 true}
!76 = !{!"WaEnableALTModeVisaWA", i1 false}
!77 = !{!"NewSpillCostFunction", i1 false}
!78 = !{!"ForceLargeGRFNum4RQ", i1 false}
!79 = !{!"DisableEUFusion", i1 false}
!80 = !{!"DisableFDivToFMulInvOpt", i1 false}
!81 = !{!"initializePhiSampleSourceWA", i1 false}
!82 = !{!"WaDisableSubspanUseNoMaskForCB", i1 false}
!83 = !{!"DisableLoosenSimd32Occu", i1 false}
!84 = !{!"FastestS1Options", i32 0}
!85 = !{!"EnableFastestForWaveIntrinsicsCS", i1 false}
!86 = !{!"FuncMD"}
!87 = !{!"pushInfo", !88, !89, !90, !94, !95, !96, !97, !98, !99, !100, !101, !114, !115, !116, !117}
!88 = !{!"pushableAddresses"}
!89 = !{!"bindlessPushInfo"}
!90 = !{!"dynamicBufferInfo", !91, !92, !93}
!91 = !{!"firstIndex", i32 0}
!92 = !{!"numOffsets", i32 0}
!93 = !{!"forceDisabled", i1 false}
!94 = !{!"MaxNumberOfPushedBuffers", i32 0}
!95 = !{!"inlineConstantBufferSlot", i32 -1}
!96 = !{!"inlineConstantBufferOffset", i32 -1}
!97 = !{!"inlineConstantBufferGRFOffset", i32 -1}
!98 = !{!"constants"}
!99 = !{!"inputs"}
!100 = !{!"constantReg"}
!101 = !{!"simplePushInfoArr", !102, !111, !112, !113}
!102 = !{!"simplePushInfoArrVec[0]", !103, !104, !105, !106, !107, !108, !109, !110}
!103 = !{!"cbIdx", i32 0}
!104 = !{!"pushableAddressGrfOffset", i32 -1}
!105 = !{!"pushableOffsetGrfOffset", i32 -1}
!106 = !{!"offset", i32 0}
!107 = !{!"size", i32 0}
!108 = !{!"isStateless", i1 false}
!109 = !{!"isBindless", i1 false}
!110 = !{!"simplePushLoads"}
!111 = !{!"simplePushInfoArrVec[1]", !103, !104, !105, !106, !107, !108, !109, !110}
!112 = !{!"simplePushInfoArrVec[2]", !103, !104, !105, !106, !107, !108, !109, !110}
!113 = !{!"simplePushInfoArrVec[3]", !103, !104, !105, !106, !107, !108, !109, !110}
!114 = !{!"simplePushBufferUsed", i32 0}
!115 = !{!"pushAnalysisWIInfos"}
!116 = !{!"inlineRTGlobalPtrOffset", i32 0}
!117 = !{!"rtSyncSurfPtrOffset", i32 0}
!118 = !{!"psInfo", !119, !120, !121, !122, !123, !124, !125, !126, !127, !128, !129, !130, !131, !132, !133}
!119 = !{!"BlendStateDisabledMask", i8 0}
!120 = !{!"SkipSrc0Alpha", i1 false}
!121 = !{!"DualSourceBlendingDisabled", i1 false}
!122 = !{!"ForceEnableSimd32", i1 false}
!123 = !{!"outputDepth", i1 false}
!124 = !{!"outputStencil", i1 false}
!125 = !{!"outputMask", i1 false}
!126 = !{!"blendToFillEnabled", i1 false}
!127 = !{!"forceEarlyZ", i1 false}
!128 = !{!"hasVersionedLoop", i1 false}
!129 = !{!"forceSingleSourceRTWAfterDualSourceRTW", i1 false}
!130 = !{!"NumSamples", i8 0}
!131 = !{!"blendOptimizationMode"}
!132 = !{!"colorOutputMask"}
!133 = !{!"WaDisableVRS", i1 false}
!134 = !{!"csInfo", !135, !136, !137, !138, !139, !21, !22, !140, !141, !142, !143, !144, !145, !146, !147, !148, !149, !150, !151, !52, !53, !152, !153, !154}
!135 = !{!"maxWorkGroupSize", i32 0}
!136 = !{!"waveSize", i32 0}
!137 = !{!"ComputeShaderSecondCompile"}
!138 = !{!"forcedSIMDSize", i8 0}
!139 = !{!"forceTotalGRFNum", i32 0}
!140 = !{!"forceSpillCompression", i1 false}
!141 = !{!"allowLowerSimd", i1 false}
!142 = !{!"disableSimd32Slicing", i1 false}
!143 = !{!"disableSplitOnSpill", i1 false}
!144 = !{!"enableNewSpillCostFunction", i1 false}
!145 = !{!"forcedVISAPreRAScheduler", i1 false}
!146 = !{!"forceUniformBuffer", i1 false}
!147 = !{!"forceUniformSurfaceSampler", i1 false}
!148 = !{!"disableLocalIdOrderOptimizations", i1 false}
!149 = !{!"disableDispatchAlongY", i1 false}
!150 = !{!"neededThreadIdLayout", i1* null}
!151 = !{!"forceTileYWalk", i1 false}
!152 = !{!"walkOrderEnabled", i1 false}
!153 = !{!"walkOrderOverride", i32 0}
!154 = !{!"ResForHfPacking"}
!155 = !{!"msInfo", !156, !157, !158, !159, !160, !161, !162, !163, !164}
!156 = !{!"PrimitiveTopology", i32 3}
!157 = !{!"MaxNumOfPrimitives", i32 0}
!158 = !{!"MaxNumOfVertices", i32 0}
!159 = !{!"MaxNumOfPerPrimitiveOutputs", i32 0}
!160 = !{!"MaxNumOfPerVertexOutputs", i32 0}
!161 = !{!"WorkGroupSize", i32 0}
!162 = !{!"WorkGroupMemorySizeInBytes", i32 0}
!163 = !{!"IndexFormat", i32 6}
!164 = !{!"SubgroupSize", i32 0}
!165 = !{!"taskInfo", !166, !161, !162, !164}
!166 = !{!"MaxNumOfOutputs", i32 0}
!167 = !{!"NBarrierCnt", i32 0}
!168 = !{!"rtInfo", !169, !170, !171, !172, !173, !174, !175, !176, !177, !178, !179, !180, !181}
!169 = !{!"RayQueryAllocSizeInBytes", i32 0}
!170 = !{!"NumContinuations", i32 0}
!171 = !{!"RTAsyncStackAddrspace", i32 -1}
!172 = !{!"RTAsyncStackSurfaceStateOffset", i1* null}
!173 = !{!"SWHotZoneAddrspace", i32 -1}
!174 = !{!"SWHotZoneSurfaceStateOffset", i1* null}
!175 = !{!"SWStackAddrspace", i32 -1}
!176 = !{!"SWStackSurfaceStateOffset", i1* null}
!177 = !{!"RTSyncStackAddrspace", i32 -1}
!178 = !{!"RTSyncStackSurfaceStateOffset", i1* null}
!179 = !{!"doSyncDispatchRays", i1 false}
!180 = !{!"MemStyle", !"Xe"}
!181 = !{!"GlobalDataStyle", !"Xe"}
!182 = !{!"CurUniqueIndirectIdx", i32 0}
!183 = !{!"inlineDynTextures"}
!184 = !{!"inlineResInfoData"}
!185 = !{!"immConstant", !186, !187, !188}
!186 = !{!"data"}
!187 = !{!"sizes"}
!188 = !{!"zeroIdxs"}
!189 = !{!"stringConstants"}
!190 = !{!"inlineBuffers", !191, !195, !196}
!191 = !{!"inlineBuffersVec[0]", !192, !193, !194}
!192 = !{!"alignment", i32 0}
!193 = !{!"allocSize", i64 0}
!194 = !{!"Buffer"}
!195 = !{!"inlineBuffersVec[1]", !192, !193, !194}
!196 = !{!"inlineBuffersVec[2]", !192, !193, !194}
!197 = !{!"GlobalPointerProgramBinaryInfos"}
!198 = !{!"ConstantPointerProgramBinaryInfos"}
!199 = !{!"GlobalBufferAddressRelocInfo"}
!200 = !{!"ConstantBufferAddressRelocInfo"}
!201 = !{!"forceLscCacheList"}
!202 = !{!"SrvMap"}
!203 = !{!"RasterizerOrderedByteAddressBuffer"}
!204 = !{!"RasterizerOrderedViews"}
!205 = !{!"MinNOSPushConstantSize", i32 0}
!206 = !{!"inlineProgramScopeOffsets"}
!207 = !{!"shaderData", !208}
!208 = !{!"numReplicas", i32 0}
!209 = !{!"URBInfo", !210, !211, !212}
!210 = !{!"has64BVertexHeaderInput", i1 false}
!211 = !{!"has64BVertexHeaderOutput", i1 false}
!212 = !{!"hasVertexHeader", i1 true}
!213 = !{!"UseBindlessImage", i1 false}
!214 = !{!"enableRangeReduce", i1 false}
!215 = !{!"allowMatchMadOptimizationforVS", i1 false}
!216 = !{!"disableMatchMadOptimizationForCS", i1 false}
!217 = !{!"disableMemOptforNegativeOffsetLoads", i1 false}
!218 = !{!"enableThreeWayLoadSpiltOpt", i1 false}
!219 = !{!"statefulResourcesNotAliased", i1 false}
!220 = !{!"disableMixMode", i1 false}
!221 = !{!"genericAccessesResolved", i1 false}
!222 = !{!"disableSeparateSpillPvtScratchSpace", i1 false}
!223 = !{!"disableSeparateScratchWA", i1 false}
!224 = !{!"privateMemoryPerWI", i32 0}
!225 = !{!"PrivateMemoryPerFG"}
!226 = !{!"m_OptsToDisable"}
!227 = !{!"capabilities", !228}
!228 = !{!"globalVariableDecorationsINTEL", i1 false}
!229 = !{!"m_ShaderResourceViewMcsMask", !230, !231}
!230 = !{!"m_ShaderResourceViewMcsMaskVec[0]", i64 0}
!231 = !{!"m_ShaderResourceViewMcsMaskVec[1]", i64 0}
!232 = !{!"computedDepthMode", i32 0}
!233 = !{!"isHDCFastClearShader", i1 false}
!234 = !{!"argRegisterReservations", !235}
!235 = !{!"argRegisterReservationsVec[0]", i32 0}
!236 = !{i32 1, i32 0, i32 0, i32 0, i32 1, i32 0, i32 0, i32 1, i32 0, i32 0}
!237 = !{!"none", !"none", !"none", !"none", !"none", !"none", !"none", !"none", !"none", !"none"}
!238 = !{!"class.sycl::_V1::ext::oneapi::bfloat16*", !"class.sycl::_V1::range", !"class.sycl::_V1::range", !"long", !"class.sycl::_V1::ext::oneapi::bfloat16*", !"class.sycl::_V1::range", !"class.sycl::_V1::range", !"float*", !"class.sycl::_V1::range", !"class.sycl::_V1::range"}
!239 = !{!"", !"", !"", !"", !"", !"", !"", !"", !"", !""}
!240 = !{!241, !243, !243, !4, !241, !243, !243, !246, !243, !243}
!241 = !{!242}
!242 = !{i32 44, i32 2}
!243 = !{!244, !245}
!244 = !{i32 38, i32 2}
!245 = !{i32 44, i32 8}
!246 = !{!247}
!247 = !{i32 44, i32 4}
!248 = !{!245}
!249 = !{!250}
!250 = !{i32 4469}
!251 = !{!250, !252}
!252 = !{i32 4470}
