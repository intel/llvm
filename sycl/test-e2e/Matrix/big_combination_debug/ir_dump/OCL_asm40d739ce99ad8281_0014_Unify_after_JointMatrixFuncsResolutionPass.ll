; ------------------------------------------------
; OCL_asm40d739ce99ad8281_0014_Unify_after_JointMatrixFuncsResolutionPass.ll
; LLVM major version: 14
; ------------------------------------------------
target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v16:16:16-v24:32:32-v32:32:32-v48:64:64-v64:64:64-v96:128:128-v128:128:128-v192:256:256-v256:256:256-v512:512:512-v1024:1024:1024-n8:16:32"
target triple = "spir64-unknown-unknown"

%"class.sycl::_V1::ext::oneapi::bfloat16" = type { i16 }
%"class.sycl::_V1::range" = type { %"class.sycl::_V1::detail::array" }
%"class.sycl::_V1::detail::array" = type { [2 x i64] }
%spirv.JointMatrixINTEL._float_32_32_3_3_2 = type opaque
%spirv.JointMatrixINTEL._short_32_16_0_3_0 = type opaque
%spirv.JointMatrixINTEL._short_16_32_2_3_1 = type opaque

; Function Attrs: nounwind
define spir_kernel void @_ZTS7imatrixIfLm32ELm32ELm16EE(float addrspace(1)* align 4 %0, %"class.sycl::_V1::range"* byval(%"class.sycl::_V1::range") align 8 %1, %"class.sycl::_V1::range"* byval(%"class.sycl::_V1::range") align 8 %2, i64 %3, %"class.sycl::_V1::ext::oneapi::bfloat16" addrspace(1)* align 2 %4, %"class.sycl::_V1::range"* byval(%"class.sycl::_V1::range") align 8 %5, %"class.sycl::_V1::range"* byval(%"class.sycl::_V1::range") align 8 %6, %"class.sycl::_V1::ext::oneapi::bfloat16" addrspace(1)* align 2 %7, %"class.sycl::_V1::range"* byval(%"class.sycl::_V1::range") align 8 %8, %"class.sycl::_V1::range"* byval(%"class.sycl::_V1::range") align 8 %9) #0 !kernel_arg_addr_space !437 !kernel_arg_access_qual !438 !kernel_arg_type !439 !kernel_arg_type_qual !440 !kernel_arg_base_type !439 !kernel_arg_name !440 !spirv.ParameterDecorations !441 {
  %11 = alloca <32 x i32>, align 128
  %12 = alloca <32 x i32>, align 128
  %13 = alloca [2 x <64 x float>], align 256
  %14 = alloca [2 x <64 x float>], align 256
  call spir_func void @__itt_offload_wi_start_wrapper() #1
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
  %48 = call spir_func i64 @_Z33__spirv_BuiltInGlobalInvocationIdi(i32 1) #5
  %49 = icmp ult i64 %48, 2147483648
  call void @llvm.assume(i1 %49)
  %50 = call spir_func i64 @_Z33__spirv_BuiltInGlobalInvocationIdi(i32 0) #5
  %51 = icmp ult i64 %50, 2147483648
  call void @llvm.assume(i1 %51)
  %52 = call spir_func i64 @_Z32__spirv_BuiltInLocalInvocationIdi(i32 1) #5
  %53 = icmp ult i64 %52, 2147483648
  call void @llvm.assume(i1 %53)
  %54 = sub nsw i64 %48, %52, !spirv.Decorations !450
  %55 = call spir_func i64 @_Z32__spirv_BuiltInLocalInvocationIdi(i32 0) #5
  %56 = icmp ult i64 %55, 2147483648
  call void @llvm.assume(i1 %56)
  %57 = sub nsw i64 %50, %55, !spirv.Decorations !450
  %58 = add i64 %23, %22
  %59 = sub i64 0, %58
  %60 = getelementptr inbounds float, float addrspace(1)* %25, i64 %59
  %61 = shl nsw i64 %54, 11, !spirv.Decorations !450
  %62 = getelementptr inbounds float, float addrspace(1)* %60, i64 %61
  %63 = udiv i64 %57, %3
  %64 = shl i64 %63, 5
  %65 = getelementptr inbounds float, float addrspace(1)* %62, i64 %64
  %66 = bitcast [2 x <64 x float>]* %14 to i8*
  call void @__builtin_spriv_OpJointMatrixLoadINTEL_Accumulator_RowMajor_32x32_i32_128_global_v8i8_pi32_i32(i8* %66, float addrspace(1)* %65, i64 64, i32 0)
  %67 = bitcast [2 x <64 x float>]* %14 to <64 x float>*
  %68 = load <64 x float>, <64 x float>* %67, align 256
  %69 = getelementptr <64 x float>, <64 x float>* %67, i32 1
  %70 = load <64 x float>, <64 x float>* %69, align 256
  %71 = insertvalue [2 x <64 x float>] undef, <64 x float> %68, 0
  %72 = insertvalue [2 x <64 x float>] %71, <64 x float> %70, 1
  %73 = add i64 %34, %33
  %74 = sub i64 0, %73
  %75 = getelementptr inbounds %"class.sycl::_V1::ext::oneapi::bfloat16", %"class.sycl::_V1::ext::oneapi::bfloat16" addrspace(1)* %36, i64 %74
  %76 = shl nsw i64 %54, 10, !spirv.Decorations !450
  %77 = getelementptr inbounds %"class.sycl::_V1::ext::oneapi::bfloat16", %"class.sycl::_V1::ext::oneapi::bfloat16" addrspace(1)* %75, i64 %76
  %78 = add i64 %45, %44
  %79 = sub i64 0, %78
  %80 = getelementptr inbounds %"class.sycl::_V1::ext::oneapi::bfloat16", %"class.sycl::_V1::ext::oneapi::bfloat16" addrspace(1)* %47, i64 %79
  %81 = shl i64 %63, 6
  %82 = getelementptr %"class.sycl::_V1::ext::oneapi::bfloat16", %"class.sycl::_V1::ext::oneapi::bfloat16" addrspace(1)* %80, i64 %81
  br label %83

83:                                               ; preds = %86, %10
  %84 = phi i32 [ 0, %10 ], [ %96, %86 ]
  %85 = icmp ult i32 %84, 2
  br i1 %85, label %86, label %97

86:                                               ; preds = %83
  %87 = shl nuw nsw i32 %84, 4, !spirv.Decorations !452
  %88 = zext i32 %87 to i64
  %89 = getelementptr inbounds %"class.sycl::_V1::ext::oneapi::bfloat16", %"class.sycl::_V1::ext::oneapi::bfloat16" addrspace(1)* %77, i64 %88
  %90 = bitcast <32 x i32>* %12 to i8*
  call void @__builtin_spriv_OpJointMatrixLoadINTEL_PackedA_RowMajor_32x16_i16_32_global_v8i8_pi32_i32(i8* %90, %"class.sycl::_V1::ext::oneapi::bfloat16" addrspace(1)* %89, i64 32, i32 0)
  %91 = load <32 x i32>, <32 x i32>* %12, align 128
  %92 = shl nuw nsw i64 %88, 6, !spirv.Decorations !452
  %93 = getelementptr %"class.sycl::_V1::ext::oneapi::bfloat16", %"class.sycl::_V1::ext::oneapi::bfloat16" addrspace(1)* %82, i64 %92
  %94 = bitcast <32 x i32>* %11 to i8*
  call void @__builtin_spriv_OpJointMatrixLoadINTEL_PackedB_PackedB_16x32_i16_32_global_v8i8_pi32_i32(i8* %94, %"class.sycl::_V1::ext::oneapi::bfloat16" addrspace(1)* %93, i64 128, i32 0)
  %95 = load <32 x i32>, <32 x i32>* %11, align 128
  %96 = add nuw nsw i32 %84, 1, !spirv.Decorations !452
  br label %83

97:                                               ; preds = %83
  store [2 x <64 x float>] %72, [2 x <64 x float>]* %13, align 256
  %98 = bitcast [2 x <64 x float>]* %13 to i8*
  call void @__builtin_spriv_OpJointMatrixStoreINTEL_Accumulator_RowMajor_32x32_i32_128_global_pi64_v8i8(float addrspace(1)* %65, i8* %98, i64 64, i32 0)
  call spir_func void @__itt_offload_wi_finish_wrapper() #1
  ret void
}

; Function Attrs: alwaysinline nounwind
define spir_func void @__itt_offload_wi_start_wrapper() #1 {
  %1 = alloca [3 x i64], align 8, !spirv.Decorations !454
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
  %4 = alloca i64 addrspace(4)*, align 8, !spirv.Decorations !454
  %5 = alloca i64, align 8, !spirv.Decorations !454
  %6 = alloca i32, align 4, !spirv.Decorations !442
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
declare spir_func %spirv.JointMatrixINTEL._float_32_32_3_3_2 addrspace(1)* @_Z81__spirv_JointMatrixLoadINTEL_RPU3AS143__spirv_JointMatrixINTEL__float_32_32_3_3_2PU3AS1fliii(float addrspace(1)*, i64, i32, i32, i32) #0

; Function Attrs: nounwind
declare spir_func %spirv.JointMatrixINTEL._short_32_16_0_3_0 addrspace(1)* @"_Z81__spirv_JointMatrixLoadINTEL_RPU3AS143__spirv_JointMatrixINTEL__short_32_16_0_3_0PU3AS138class.sycl::_V1::ext::oneapi::bfloat16liii"(%"class.sycl::_V1::ext::oneapi::bfloat16" addrspace(1)*, i64, i32, i32, i32) #0

; Function Attrs: nounwind
declare spir_func %spirv.JointMatrixINTEL._short_16_32_2_3_1 addrspace(1)* @"_Z81__spirv_JointMatrixLoadINTEL_RPU3AS143__spirv_JointMatrixINTEL__short_16_32_2_3_1PU3AS138class.sycl::_V1::ext::oneapi::bfloat16liii"(%"class.sycl::_V1::ext::oneapi::bfloat16" addrspace(1)*, i64, i32, i32, i32) #0

; Function Attrs: nounwind
declare spir_func void @_Z29__spirv_JointMatrixStoreINTELPU3AS1fPU3AS143__spirv_JointMatrixINTEL__float_32_32_3_3_2liii(float addrspace(1)*, %spirv.JointMatrixINTEL._float_32_32_3_3_2 addrspace(1)*, i64, i32, i32, i32) #0

; Function Attrs: alwaysinline nounwind
define spir_func void @__itt_offload_wi_finish_wrapper() #1 {
  %1 = alloca [3 x i64], align 8, !spirv.Decorations !454
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
  %3 = alloca i64 addrspace(4)*, align 8, !spirv.Decorations !454
  %4 = alloca i64, align 8, !spirv.Decorations !454
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

declare void @__builtin_spriv_OpJointMatrixLoadINTEL_Accumulator_RowMajor_32x32_i32_128_global_v8i8_pi32_i32(i8*, float addrspace(1)*, i64, i32)

declare void @__builtin_spriv_OpJointMatrixStoreINTEL_Accumulator_RowMajor_32x32_i32_128_global_pi64_v8i8(float addrspace(1)*, i8*, i64, i32)

declare void @__builtin_spriv_OpJointMatrixLoadINTEL_PackedA_RowMajor_32x16_i16_32_global_v8i8_pi32_i32(i8*, %"class.sycl::_V1::ext::oneapi::bfloat16" addrspace(1)*, i64, i32)

declare void @__builtin_spriv_OpJointMatrixLoadINTEL_PackedB_PackedB_16x32_i16_32_global_v8i8_pi32_i32(i8*, %"class.sycl::_V1::ext::oneapi::bfloat16" addrspace(1)*, i64, i32)

attributes #0 = { nounwind }
attributes #1 = { alwaysinline nounwind }
attributes #2 = { argmemonly nofree nosync nounwind willreturn }
attributes #3 = { noinline nounwind optnone }
attributes #4 = { inaccessiblememonly nofree nosync nounwind willreturn }
attributes #5 = { nounwind readnone willreturn }

!spirv.MemoryModel = !{!0}
!spirv.Source = !{!1}
!spirv.Generator = !{!2}
!igc.functions = !{!3}
!IGCMetadata = !{!7}

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
!112 = !{!"dim1", i32 0}
!113 = !{!"dim2", i32 0}
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
!437 = !{i32 1, i32 0, i32 0, i32 0, i32 1, i32 0, i32 0, i32 1, i32 0, i32 0}
!438 = !{!"none", !"none", !"none", !"none", !"none", !"none", !"none", !"none", !"none", !"none"}
!439 = !{!"float*", !"class.sycl::_V1::range", !"class.sycl::_V1::range", !"long", !"class.sycl::_V1::ext::oneapi::bfloat16*", !"class.sycl::_V1::range", !"class.sycl::_V1::range", !"class.sycl::_V1::ext::oneapi::bfloat16*", !"class.sycl::_V1::range", !"class.sycl::_V1::range"}
!440 = !{!"", !"", !"", !"", !"", !"", !"", !"", !"", !""}
!441 = !{!442, !444, !444, !447, !448, !444, !444, !448, !444, !444}
!442 = !{!443}
!443 = !{i32 44, i32 4}
!444 = !{!445, !446}
!445 = !{i32 38, i32 2}
!446 = !{i32 44, i32 8}
!447 = !{}
!448 = !{!449}
!449 = !{i32 44, i32 2}
!450 = !{!451}
!451 = !{i32 4469}
!452 = !{!451, !453}
!453 = !{i32 4470}
!454 = !{!446}
