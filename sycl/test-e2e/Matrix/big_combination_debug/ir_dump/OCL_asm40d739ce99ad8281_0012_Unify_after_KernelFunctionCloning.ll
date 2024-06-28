; ------------------------------------------------
; OCL_asm40d739ce99ad8281_0012_Unify_after_KernelFunctionCloning.ll
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
define spir_kernel void @_ZTS7imatrixIfLm32ELm32ELm16EE(float addrspace(1)* align 4 %0, %"class.sycl::_V1::range"* byval(%"class.sycl::_V1::range") align 8 %1, %"class.sycl::_V1::range"* byval(%"class.sycl::_V1::range") align 8 %2, i64 %3, %"class.sycl::_V1::ext::oneapi::bfloat16" addrspace(1)* align 2 %4, %"class.sycl::_V1::range"* byval(%"class.sycl::_V1::range") align 8 %5, %"class.sycl::_V1::range"* byval(%"class.sycl::_V1::range") align 8 %6, %"class.sycl::_V1::ext::oneapi::bfloat16" addrspace(1)* align 2 %7, %"class.sycl::_V1::range"* byval(%"class.sycl::_V1::range") align 8 %8, %"class.sycl::_V1::range"* byval(%"class.sycl::_V1::range") align 8 %9) #0 !kernel_arg_addr_space !436 !kernel_arg_access_qual !437 !kernel_arg_type !438 !kernel_arg_type_qual !439 !kernel_arg_base_type !438 !kernel_arg_name !439 !spirv.ParameterDecorations !440 {
  call spir_func void @__itt_offload_wi_start_wrapper() #1
  %11 = bitcast %"class.sycl::_V1::range"* %1 to i64*
  %12 = getelementptr inbounds i64, i64* %11, i64 1
  %13 = load i64, i64* %12, align 8
  %14 = bitcast %"class.sycl::_V1::range"* %2 to i64*
  %15 = load i64, i64* %14, align 8
  %16 = bitcast %"class.sycl::_V1::range"* %2 to i64*
  %17 = getelementptr inbounds i64, i64* %16, i64 1
  %18 = load i64, i64* %17, align 8
  %19 = mul i64 %15, %13
  %20 = getelementptr float, float addrspace(1)* %0, i64 %19
  %21 = getelementptr float, float addrspace(1)* %20, i64 %18
  %22 = bitcast %"class.sycl::_V1::range"* %5 to i64*
  %23 = getelementptr inbounds i64, i64* %22, i64 1
  %24 = load i64, i64* %23, align 8
  %25 = bitcast %"class.sycl::_V1::range"* %6 to i64*
  %26 = load i64, i64* %25, align 8
  %27 = bitcast %"class.sycl::_V1::range"* %6 to i64*
  %28 = getelementptr inbounds i64, i64* %27, i64 1
  %29 = load i64, i64* %28, align 8
  %30 = mul i64 %26, %24
  %31 = getelementptr %"class.sycl::_V1::ext::oneapi::bfloat16", %"class.sycl::_V1::ext::oneapi::bfloat16" addrspace(1)* %4, i64 %30
  %32 = getelementptr %"class.sycl::_V1::ext::oneapi::bfloat16", %"class.sycl::_V1::ext::oneapi::bfloat16" addrspace(1)* %31, i64 %29
  %33 = bitcast %"class.sycl::_V1::range"* %8 to i64*
  %34 = getelementptr inbounds i64, i64* %33, i64 1
  %35 = load i64, i64* %34, align 8
  %36 = bitcast %"class.sycl::_V1::range"* %9 to i64*
  %37 = load i64, i64* %36, align 8
  %38 = bitcast %"class.sycl::_V1::range"* %9 to i64*
  %39 = getelementptr inbounds i64, i64* %38, i64 1
  %40 = load i64, i64* %39, align 8
  %41 = mul i64 %37, %35
  %42 = getelementptr %"class.sycl::_V1::ext::oneapi::bfloat16", %"class.sycl::_V1::ext::oneapi::bfloat16" addrspace(1)* %7, i64 %41
  %43 = getelementptr %"class.sycl::_V1::ext::oneapi::bfloat16", %"class.sycl::_V1::ext::oneapi::bfloat16" addrspace(1)* %42, i64 %40
  %44 = call spir_func i64 @_Z33__spirv_BuiltInGlobalInvocationIdi(i32 1) #5
  %45 = icmp ult i64 %44, 2147483648
  call void @llvm.assume(i1 %45)
  %46 = call spir_func i64 @_Z33__spirv_BuiltInGlobalInvocationIdi(i32 0) #5
  %47 = icmp ult i64 %46, 2147483648
  call void @llvm.assume(i1 %47)
  %48 = call spir_func i64 @_Z32__spirv_BuiltInLocalInvocationIdi(i32 1) #5
  %49 = icmp ult i64 %48, 2147483648
  call void @llvm.assume(i1 %49)
  %50 = sub nsw i64 %44, %48, !spirv.Decorations !449
  %51 = call spir_func i64 @_Z32__spirv_BuiltInLocalInvocationIdi(i32 0) #5
  %52 = icmp ult i64 %51, 2147483648
  call void @llvm.assume(i1 %52)
  %53 = sub nsw i64 %46, %51, !spirv.Decorations !449
  %54 = add i64 %19, %18
  %55 = sub i64 0, %54
  %56 = getelementptr inbounds float, float addrspace(1)* %21, i64 %55
  %57 = shl nsw i64 %50, 11, !spirv.Decorations !449
  %58 = getelementptr inbounds float, float addrspace(1)* %56, i64 %57
  %59 = udiv i64 %53, %3
  %60 = shl i64 %59, 5
  %61 = getelementptr inbounds float, float addrspace(1)* %58, i64 %60
  %62 = call spir_func %spirv.JointMatrixINTEL._float_32_32_3_3_2 addrspace(1)* @_Z81__spirv_JointMatrixLoadINTEL_RPU3AS143__spirv_JointMatrixINTEL__float_32_32_3_3_2PU3AS1fliii(float addrspace(1)* %61, i64 64, i32 0, i32 3, i32 0) #0
  %63 = add i64 %30, %29
  %64 = sub i64 0, %63
  %65 = getelementptr inbounds %"class.sycl::_V1::ext::oneapi::bfloat16", %"class.sycl::_V1::ext::oneapi::bfloat16" addrspace(1)* %32, i64 %64
  %66 = shl nsw i64 %50, 10, !spirv.Decorations !449
  %67 = getelementptr inbounds %"class.sycl::_V1::ext::oneapi::bfloat16", %"class.sycl::_V1::ext::oneapi::bfloat16" addrspace(1)* %65, i64 %66
  %68 = add i64 %41, %40
  %69 = sub i64 0, %68
  %70 = getelementptr inbounds %"class.sycl::_V1::ext::oneapi::bfloat16", %"class.sycl::_V1::ext::oneapi::bfloat16" addrspace(1)* %43, i64 %69
  %71 = shl i64 %59, 6
  %72 = getelementptr %"class.sycl::_V1::ext::oneapi::bfloat16", %"class.sycl::_V1::ext::oneapi::bfloat16" addrspace(1)* %70, i64 %71
  br label %73

73:                                               ; preds = %76, %10
  %74 = phi i32 [ 0, %10 ], [ %84, %76 ]
  %75 = icmp ult i32 %74, 2
  br i1 %75, label %76, label %85

76:                                               ; preds = %73
  %77 = shl nuw nsw i32 %74, 4, !spirv.Decorations !451
  %78 = zext i32 %77 to i64
  %79 = getelementptr inbounds %"class.sycl::_V1::ext::oneapi::bfloat16", %"class.sycl::_V1::ext::oneapi::bfloat16" addrspace(1)* %67, i64 %78
  %80 = call spir_func %spirv.JointMatrixINTEL._short_32_16_0_3_0 addrspace(1)* @"_Z81__spirv_JointMatrixLoadINTEL_RPU3AS143__spirv_JointMatrixINTEL__short_32_16_0_3_0PU3AS138class.sycl::_V1::ext::oneapi::bfloat16liii"(%"class.sycl::_V1::ext::oneapi::bfloat16" addrspace(1)* %79, i64 32, i32 0, i32 3, i32 0) #0
  %81 = shl nuw nsw i64 %78, 6, !spirv.Decorations !451
  %82 = getelementptr %"class.sycl::_V1::ext::oneapi::bfloat16", %"class.sycl::_V1::ext::oneapi::bfloat16" addrspace(1)* %72, i64 %81
  %83 = call spir_func %spirv.JointMatrixINTEL._short_16_32_2_3_1 addrspace(1)* @"_Z81__spirv_JointMatrixLoadINTEL_RPU3AS143__spirv_JointMatrixINTEL__short_16_32_2_3_1PU3AS138class.sycl::_V1::ext::oneapi::bfloat16liii"(%"class.sycl::_V1::ext::oneapi::bfloat16" addrspace(1)* %82, i64 128, i32 2, i32 3, i32 0) #0
  %84 = add nuw nsw i32 %74, 1, !spirv.Decorations !451
  br label %73

85:                                               ; preds = %73
  call spir_func void @_Z29__spirv_JointMatrixStoreINTELPU3AS1fPU3AS143__spirv_JointMatrixINTEL__float_32_32_3_3_2liii(float addrspace(1)* %61, %spirv.JointMatrixINTEL._float_32_32_3_3_2 addrspace(1)* %62, i64 64, i32 0, i32 3, i32 0) #0
  call spir_func void @__itt_offload_wi_finish_wrapper() #1
  ret void
}

; Function Attrs: alwaysinline nounwind
define spir_func void @__itt_offload_wi_start_wrapper() #1 {
  %1 = alloca [3 x i64], align 8, !spirv.Decorations !453
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
  %4 = alloca i64 addrspace(4)*, align 8, !spirv.Decorations !453
  %5 = alloca i64, align 8, !spirv.Decorations !453
  %6 = alloca i32, align 4, !spirv.Decorations !441
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
  %1 = alloca [3 x i64], align 8, !spirv.Decorations !453
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
  %3 = alloca i64 addrspace(4)*, align 8, !spirv.Decorations !453
  %4 = alloca i64, align 8, !spirv.Decorations !453
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
!spirv.Source = !{!1}
!spirv.Generator = !{!2}
!igc.functions = !{!3}
!IGCMetadata = !{!6}

!0 = !{i32 2, i32 2}
!1 = !{i32 4, i32 100000}
!2 = !{i16 6, i16 14}
!3 = !{void (float addrspace(1)*, %"class.sycl::_V1::range"*, %"class.sycl::_V1::range"*, i64, %"class.sycl::_V1::ext::oneapi::bfloat16" addrspace(1)*, %"class.sycl::_V1::range"*, %"class.sycl::_V1::range"*, %"class.sycl::_V1::ext::oneapi::bfloat16" addrspace(1)*, %"class.sycl::_V1::range"*, %"class.sycl::_V1::range"*)* @_ZTS7imatrixIfLm32ELm32ELm16EE, !4}
!4 = !{!5}
!5 = !{!"function_type", i32 0}
!6 = !{!"ModuleMD", !7, !8, !105, !231, !262, !263, !267, !270, !271, !272, !307, !333, !346, !347, !348, !363, !364, !365, !366, !367, !368, !369, !370, !371, !372, !376, !377, !384, !385, !386, !387, !388, !389, !390, !391, !392, !393, !394, !395, !397, !401, !402, !403, !404, !405, !406, !407, !408, !409, !410, !411, !412, !413, !414, !415, !416, !417, !155, !418, !419, !420, !422, !425, !426, !427, !429, !430, !431}
!7 = !{!"isPrecise", i1 false}
!8 = !{!"compOpt", !9, !10, !11, !12, !13, !14, !15, !16, !17, !18, !19, !20, !21, !22, !23, !24, !25, !26, !27, !28, !29, !30, !31, !32, !33, !34, !35, !36, !37, !38, !39, !40, !41, !42, !43, !44, !45, !46, !47, !48, !49, !50, !51, !52, !53, !54, !55, !56, !57, !58, !59, !60, !61, !62, !63, !64, !65, !66, !67, !68, !69, !70, !71, !72, !73, !74, !75, !76, !77, !78, !79, !80, !81, !82, !83, !84, !85, !86, !87, !88, !89, !90, !91, !92, !93, !94, !95, !96, !97, !98, !99, !100, !101, !102, !103, !104}
!9 = !{!"DenormsAreZero", i1 false}
!10 = !{!"BFTFDenormsAreZero", i1 false}
!11 = !{!"CorrectlyRoundedDivSqrt", i1 false}
!12 = !{!"OptDisable", i1 false}
!13 = !{!"MadEnable", i1 true}
!14 = !{!"NoSignedZeros", i1 false}
!15 = !{!"NoNaNs", i1 false}
!16 = !{!"FloatRoundingMode", i32 0}
!17 = !{!"FloatCvtIntRoundingMode", i32 3}
!18 = !{!"LoadCacheDefault", i32 4}
!19 = !{!"StoreCacheDefault", i32 7}
!20 = !{!"VISAPreSchedRPThreshold", i32 0}
!21 = !{!"SetLoopUnrollThreshold", i32 0}
!22 = !{!"UnsafeMathOptimizations", i1 false}
!23 = !{!"disableCustomUnsafeOpts", i1 false}
!24 = !{!"disableReducePow", i1 false}
!25 = !{!"disableSqrtOpt", i1 false}
!26 = !{!"FiniteMathOnly", i1 false}
!27 = !{!"FastRelaxedMath", i1 false}
!28 = !{!"DashGSpecified", i1 false}
!29 = !{!"FastCompilation", i1 false}
!30 = !{!"UseScratchSpacePrivateMemory", i1 true}
!31 = !{!"RelaxedBuiltins", i1 false}
!32 = !{!"SubgroupIndependentForwardProgressRequired", i1 true}
!33 = !{!"GreaterThan2GBBufferRequired", i1 true}
!34 = !{!"GreaterThan4GBBufferRequired", i1 false}
!35 = !{!"DisableA64WA", i1 false}
!36 = !{!"ForceEnableA64WA", i1 false}
!37 = !{!"PushConstantsEnable", i1 true}
!38 = !{!"HasPositivePointerOffset", i1 false}
!39 = !{!"HasBufferOffsetArg", i1 true}
!40 = !{!"BufferOffsetArgOptional", i1 true}
!41 = !{!"replaceGlobalOffsetsByZero", i1 false}
!42 = !{!"forcePixelShaderSIMDMode", i32 0}
!43 = !{!"ForceGeomFFShaderSIMDMode", i32 0}
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
!54 = !{!"DisableEarlyOut", i1 false}
!55 = !{!"ForceInt32DivRemEmu", i1 false}
!56 = !{!"ForceInt32DivRemEmuSP", i1 false}
!57 = !{!"WaveIntrinsicUsed", i1 false}
!58 = !{!"DisableMultiPolyPS", i1 false}
!59 = !{!"NeedTexture3DLODWA", i1 false}
!60 = !{!"DisableFastestSingleCSSIMD", i1 false}
!61 = !{!"DisableFastestLinearScan", i1 false}
!62 = !{!"UseStatelessforPrivateMemory", i1 false}
!63 = !{!"EnableTakeGlobalAddress", i1 false}
!64 = !{!"IsLibraryCompilation", i1 false}
!65 = !{!"LibraryCompileSIMDSize", i32 0}
!66 = !{!"FastVISACompile", i1 false}
!67 = !{!"MatchSinCosPi", i1 false}
!68 = !{!"ExcludeIRFromZEBinary", i1 false}
!69 = !{!"EmitZeBinVISASections", i1 false}
!70 = !{!"FP64GenEmulationEnabled", i1 false}
!71 = !{!"FP64GenConvEmulationEnabled", i1 false}
!72 = !{!"allowDisableRematforCS", i1 false}
!73 = !{!"DisableIncSpillCostAllAddrTaken", i1 false}
!74 = !{!"DisableCPSOmaskWA", i1 false}
!75 = !{!"DisableFastestGopt", i1 false}
!76 = !{!"WaForceHalfPromotionComputeShader", i1 false}
!77 = !{!"WaForceHalfPromotionPixelVertexShader", i1 false}
!78 = !{!"DisableConstantCoalescing", i1 false}
!79 = !{!"EnableUndefAlphaOutputAsRed", i1 true}
!80 = !{!"WaEnableALTModeVisaWA", i1 false}
!81 = !{!"WaEnableAtomicWaveFusion", i1 false}
!82 = !{!"WaEnableAtomicWaveFusionNonNullResource", i1 false}
!83 = !{!"WaEnableAtomicWaveFusionStateless", i1 false}
!84 = !{!"WaEnableAtomicWaveFusionTyped", i1 false}
!85 = !{!"ForceCBThroughSampler3D", i1 false}
!86 = !{!"WaStoreRawVectorToTypedWrite", i1 false}
!87 = !{!"WaLoadRawVectorToTypedRead", i1 false}
!88 = !{!"WaZeroSLMBeforeUse", i1 false}
!89 = !{!"WaFlagGroupTypedUAVGloballyCoherent", i1 false}
!90 = !{!"NewSpillCostFunction", i1 false}
!91 = !{!"EnableVRT", i1 false}
!92 = !{!"ForceLargeGRFNum4RQ", i1 false}
!93 = !{!"Enable2xGRFRetry", i1 false}
!94 = !{!"Detect2xGRFCandidate", i1 false}
!95 = !{!"EnableURBWritesMerging", i1 true}
!96 = !{!"DisableEUFusion", i1 false}
!97 = !{!"DisableFDivToFMulInvOpt", i1 false}
!98 = !{!"initializePhiSampleSourceWA", i1 false}
!99 = !{!"WaDisableSubspanUseNoMaskForCB", i1 false}
!100 = !{!"DisableLoosenSimd32Occu", i1 false}
!101 = !{!"FastestS1Options", i32 0}
!102 = !{!"DisableFastestForWaveIntrinsicsCS", i1 false}
!103 = !{!"ForceLinearWalkOnLinearUAV", i1 false}
!104 = !{!"DisableLscSamplerRouting", i1 false}
!105 = !{!"FuncMD", !106, !107}
!106 = !{!"FuncMDMap[0]", void (float addrspace(1)*, %"class.sycl::_V1::range"*, %"class.sycl::_V1::range"*, i64, %"class.sycl::_V1::ext::oneapi::bfloat16" addrspace(1)*, %"class.sycl::_V1::range"*, %"class.sycl::_V1::range"*, %"class.sycl::_V1::ext::oneapi::bfloat16" addrspace(1)*, %"class.sycl::_V1::range"*, %"class.sycl::_V1::range"*)* @_ZTS7imatrixIfLm32ELm32ELm16EE}
!107 = !{!"FuncMDValue[0]", !108, !109, !113, !114, !115, !116, !117, !118, !119, !141, !147, !148, !149, !150, !151, !152, !153, !154, !155, !156, !157, !158, !159, !160, !161, !162, !163, !174, !185, !196, !207, !218, !229, !230}
!108 = !{!"localOffsets"}
!109 = !{!"workGroupWalkOrder", !110, !111, !112}
!110 = !{!"dim0", i32 0}
!111 = !{!"dim1", i32 0}
!112 = !{!"dim2", i32 0}
!113 = !{!"funcArgs"}
!114 = !{!"functionType", !"KernelFunction"}
!115 = !{!"inlineDynConstants"}
!116 = !{!"inlineDynRootConstant"}
!117 = !{!"inlineDynConstantDescTable"}
!118 = !{!"m_pInterestingConstants"}
!119 = !{!"rtInfo", !120, !121, !122, !123, !124, !125, !126, !127, !128, !129, !130, !131, !132, !133, !134, !135, !139, !140, !91}
!120 = !{!"callableShaderType", !"NumberOfCallableShaderTypes"}
!121 = !{!"isContinuation", i1 false}
!122 = !{!"hasTraceRayPayload", i1 false}
!123 = !{!"hasHitAttributes", i1 false}
!124 = !{!"hasCallableData", i1 false}
!125 = !{!"ShaderStackSize", i32 0}
!126 = !{!"ShaderHash", i64 0}
!127 = !{!"ShaderName", !""}
!128 = !{!"ParentName", !""}
!129 = !{!"SlotNum", i1* null}
!130 = !{!"NOSSize", i32 0}
!131 = !{!"globalRootSignatureSize", i32 0}
!132 = !{!"Entries"}
!133 = !{!"SpillUnions"}
!134 = !{!"CustomHitAttrSizeInBytes", i32 0}
!135 = !{!"Types", !136, !137, !138}
!136 = !{!"FrameStartTys"}
!137 = !{!"ArgumentTys"}
!138 = !{!"FullFrameTys"}
!139 = !{!"Aliases"}
!140 = !{!"NumGRF", i32 0}
!141 = !{!"resAllocMD", !142, !143, !144, !145, !146}
!142 = !{!"uavsNumType", i32 0}
!143 = !{!"srvsNumType", i32 0}
!144 = !{!"samplersNumType", i32 0}
!145 = !{!"argAllocMDList"}
!146 = !{!"inlineSamplersMD"}
!147 = !{!"maxByteOffsets"}
!148 = !{!"IsInitializer", i1 false}
!149 = !{!"IsFinalizer", i1 false}
!150 = !{!"CompiledSubGroupsNumber", i32 0}
!151 = !{!"hasInlineVmeSamplers", i1 false}
!152 = !{!"localSize", i32 0}
!153 = !{!"localIDPresent", i1 false}
!154 = !{!"groupIDPresent", i1 false}
!155 = !{!"privateMemoryPerWI", i32 0}
!156 = !{!"prevFPOffset", i32 0}
!157 = !{!"globalIDPresent", i1 false}
!158 = !{!"hasSyncRTCalls", i1 false}
!159 = !{!"hasNonKernelArgLoad", i1 false}
!160 = !{!"hasNonKernelArgStore", i1 false}
!161 = !{!"hasNonKernelArgAtomic", i1 false}
!162 = !{!"UserAnnotations"}
!163 = !{!"m_OpenCLArgAddressSpaces", !164, !165, !166, !167, !168, !169, !170, !171, !172, !173}
!164 = !{!"m_OpenCLArgAddressSpacesVec[0]", i32 1}
!165 = !{!"m_OpenCLArgAddressSpacesVec[1]", i32 0}
!166 = !{!"m_OpenCLArgAddressSpacesVec[2]", i32 0}
!167 = !{!"m_OpenCLArgAddressSpacesVec[3]", i32 0}
!168 = !{!"m_OpenCLArgAddressSpacesVec[4]", i32 1}
!169 = !{!"m_OpenCLArgAddressSpacesVec[5]", i32 0}
!170 = !{!"m_OpenCLArgAddressSpacesVec[6]", i32 0}
!171 = !{!"m_OpenCLArgAddressSpacesVec[7]", i32 1}
!172 = !{!"m_OpenCLArgAddressSpacesVec[8]", i32 0}
!173 = !{!"m_OpenCLArgAddressSpacesVec[9]", i32 0}
!174 = !{!"m_OpenCLArgAccessQualifiers", !175, !176, !177, !178, !179, !180, !181, !182, !183, !184}
!175 = !{!"m_OpenCLArgAccessQualifiersVec[0]", !"none"}
!176 = !{!"m_OpenCLArgAccessQualifiersVec[1]", !"none"}
!177 = !{!"m_OpenCLArgAccessQualifiersVec[2]", !"none"}
!178 = !{!"m_OpenCLArgAccessQualifiersVec[3]", !"none"}
!179 = !{!"m_OpenCLArgAccessQualifiersVec[4]", !"none"}
!180 = !{!"m_OpenCLArgAccessQualifiersVec[5]", !"none"}
!181 = !{!"m_OpenCLArgAccessQualifiersVec[6]", !"none"}
!182 = !{!"m_OpenCLArgAccessQualifiersVec[7]", !"none"}
!183 = !{!"m_OpenCLArgAccessQualifiersVec[8]", !"none"}
!184 = !{!"m_OpenCLArgAccessQualifiersVec[9]", !"none"}
!185 = !{!"m_OpenCLArgTypes", !186, !187, !188, !189, !190, !191, !192, !193, !194, !195}
!186 = !{!"m_OpenCLArgTypesVec[0]", !"float*"}
!187 = !{!"m_OpenCLArgTypesVec[1]", !"class.sycl::_V1::range"}
!188 = !{!"m_OpenCLArgTypesVec[2]", !"class.sycl::_V1::range"}
!189 = !{!"m_OpenCLArgTypesVec[3]", !"long"}
!190 = !{!"m_OpenCLArgTypesVec[4]", !"class.sycl::_V1::ext::oneapi::bfloat16*"}
!191 = !{!"m_OpenCLArgTypesVec[5]", !"class.sycl::_V1::range"}
!192 = !{!"m_OpenCLArgTypesVec[6]", !"class.sycl::_V1::range"}
!193 = !{!"m_OpenCLArgTypesVec[7]", !"class.sycl::_V1::ext::oneapi::bfloat16*"}
!194 = !{!"m_OpenCLArgTypesVec[8]", !"class.sycl::_V1::range"}
!195 = !{!"m_OpenCLArgTypesVec[9]", !"class.sycl::_V1::range"}
!196 = !{!"m_OpenCLArgBaseTypes", !197, !198, !199, !200, !201, !202, !203, !204, !205, !206}
!197 = !{!"m_OpenCLArgBaseTypesVec[0]", !"float*"}
!198 = !{!"m_OpenCLArgBaseTypesVec[1]", !"class.sycl::_V1::range"}
!199 = !{!"m_OpenCLArgBaseTypesVec[2]", !"class.sycl::_V1::range"}
!200 = !{!"m_OpenCLArgBaseTypesVec[3]", !"long"}
!201 = !{!"m_OpenCLArgBaseTypesVec[4]", !"class.sycl::_V1::ext::oneapi::bfloat16*"}
!202 = !{!"m_OpenCLArgBaseTypesVec[5]", !"class.sycl::_V1::range"}
!203 = !{!"m_OpenCLArgBaseTypesVec[6]", !"class.sycl::_V1::range"}
!204 = !{!"m_OpenCLArgBaseTypesVec[7]", !"class.sycl::_V1::ext::oneapi::bfloat16*"}
!205 = !{!"m_OpenCLArgBaseTypesVec[8]", !"class.sycl::_V1::range"}
!206 = !{!"m_OpenCLArgBaseTypesVec[9]", !"class.sycl::_V1::range"}
!207 = !{!"m_OpenCLArgTypeQualifiers", !208, !209, !210, !211, !212, !213, !214, !215, !216, !217}
!208 = !{!"m_OpenCLArgTypeQualifiersVec[0]", !""}
!209 = !{!"m_OpenCLArgTypeQualifiersVec[1]", !""}
!210 = !{!"m_OpenCLArgTypeQualifiersVec[2]", !""}
!211 = !{!"m_OpenCLArgTypeQualifiersVec[3]", !""}
!212 = !{!"m_OpenCLArgTypeQualifiersVec[4]", !""}
!213 = !{!"m_OpenCLArgTypeQualifiersVec[5]", !""}
!214 = !{!"m_OpenCLArgTypeQualifiersVec[6]", !""}
!215 = !{!"m_OpenCLArgTypeQualifiersVec[7]", !""}
!216 = !{!"m_OpenCLArgTypeQualifiersVec[8]", !""}
!217 = !{!"m_OpenCLArgTypeQualifiersVec[9]", !""}
!218 = !{!"m_OpenCLArgNames", !219, !220, !221, !222, !223, !224, !225, !226, !227, !228}
!219 = !{!"m_OpenCLArgNamesVec[0]", !""}
!220 = !{!"m_OpenCLArgNamesVec[1]", !""}
!221 = !{!"m_OpenCLArgNamesVec[2]", !""}
!222 = !{!"m_OpenCLArgNamesVec[3]", !""}
!223 = !{!"m_OpenCLArgNamesVec[4]", !""}
!224 = !{!"m_OpenCLArgNamesVec[5]", !""}
!225 = !{!"m_OpenCLArgNamesVec[6]", !""}
!226 = !{!"m_OpenCLArgNamesVec[7]", !""}
!227 = !{!"m_OpenCLArgNamesVec[8]", !""}
!228 = !{!"m_OpenCLArgNamesVec[9]", !""}
!229 = !{!"m_OpenCLArgScalarAsPointers"}
!230 = !{!"m_OptsToDisablePerFunc"}
!231 = !{!"pushInfo", !232, !233, !234, !238, !239, !240, !241, !242, !243, !244, !245, !258, !259, !260, !261}
!232 = !{!"pushableAddresses"}
!233 = !{!"bindlessPushInfo"}
!234 = !{!"dynamicBufferInfo", !235, !236, !237}
!235 = !{!"firstIndex", i32 0}
!236 = !{!"numOffsets", i32 0}
!237 = !{!"forceDisabled", i1 false}
!238 = !{!"MaxNumberOfPushedBuffers", i32 0}
!239 = !{!"inlineConstantBufferSlot", i32 -1}
!240 = !{!"inlineConstantBufferOffset", i32 -1}
!241 = !{!"inlineConstantBufferGRFOffset", i32 -1}
!242 = !{!"constants"}
!243 = !{!"inputs"}
!244 = !{!"constantReg"}
!245 = !{!"simplePushInfoArr", !246, !255, !256, !257}
!246 = !{!"simplePushInfoArrVec[0]", !247, !248, !249, !250, !251, !252, !253, !254}
!247 = !{!"cbIdx", i32 0}
!248 = !{!"pushableAddressGrfOffset", i32 -1}
!249 = !{!"pushableOffsetGrfOffset", i32 -1}
!250 = !{!"offset", i32 0}
!251 = !{!"size", i32 0}
!252 = !{!"isStateless", i1 false}
!253 = !{!"isBindless", i1 false}
!254 = !{!"simplePushLoads"}
!255 = !{!"simplePushInfoArrVec[1]", !247, !248, !249, !250, !251, !252, !253, !254}
!256 = !{!"simplePushInfoArrVec[2]", !247, !248, !249, !250, !251, !252, !253, !254}
!257 = !{!"simplePushInfoArrVec[3]", !247, !248, !249, !250, !251, !252, !253, !254}
!258 = !{!"simplePushBufferUsed", i32 0}
!259 = !{!"pushAnalysisWIInfos"}
!260 = !{!"inlineRTGlobalPtrOffset", i32 0}
!261 = !{!"rtSyncSurfPtrOffset", i32 0}
!262 = !{!"WaEnableICBPromotion", i1 false}
!263 = !{!"vsInfo", !264, !265, !266}
!264 = !{!"DrawIndirectBufferIndex", i32 -1}
!265 = !{!"vertexReordering", i32 -1}
!266 = !{!"MaxNumOfOutputs", i32 0}
!267 = !{!"hsInfo", !268, !269}
!268 = !{!"numPatchAttributesPatchBaseName", !""}
!269 = !{!"numVertexAttributesPatchBaseName", !""}
!270 = !{!"dsInfo", !266}
!271 = !{!"gsInfo", !266}
!272 = !{!"psInfo", !273, !274, !275, !276, !277, !278, !279, !280, !281, !282, !283, !284, !285, !286, !287, !288, !289, !290, !291, !292, !293, !294, !295, !296, !297, !298, !299, !300, !301, !302, !303, !304, !305, !306}
!273 = !{!"BlendStateDisabledMask", i8 0}
!274 = !{!"SkipSrc0Alpha", i1 false}
!275 = !{!"DualSourceBlendingDisabled", i1 false}
!276 = !{!"ForceEnableSimd32", i1 false}
!277 = !{!"outputDepth", i1 false}
!278 = !{!"outputStencil", i1 false}
!279 = !{!"outputMask", i1 false}
!280 = !{!"blendToFillEnabled", i1 false}
!281 = !{!"forceEarlyZ", i1 false}
!282 = !{!"hasVersionedLoop", i1 false}
!283 = !{!"forceSingleSourceRTWAfterDualSourceRTW", i1 false}
!284 = !{!"requestCPSizeRelevant", i1 false}
!285 = !{!"requestCPSize", i1 false}
!286 = !{!"texelMaskFastClearMode", !"Disabled"}
!287 = !{!"NumSamples", i8 0}
!288 = !{!"blendOptimizationMode"}
!289 = !{!"colorOutputMask"}
!290 = !{!"ProvokingVertexModeNosIndex", i32 0}
!291 = !{!"ProvokingVertexModeNosPatch", !""}
!292 = !{!"ProvokingVertexModeLast", !"Negative"}
!293 = !{!"VertexAttributesBypass", i1 false}
!294 = !{!"LegacyBaryAssignmentDisableLinear", i1 false}
!295 = !{!"LegacyBaryAssignmentDisableLinearNoPerspective", i1 false}
!296 = !{!"LegacyBaryAssignmentDisableLinearCentroid", i1 false}
!297 = !{!"LegacyBaryAssignmentDisableLinearNoPerspectiveCentroid", i1 false}
!298 = !{!"LegacyBaryAssignmentDisableLinearSample", i1 false}
!299 = !{!"LegacyBaryAssignmentDisableLinearNoPerspectiveSample", i1 false}
!300 = !{!"MeshShaderWAPerPrimitiveUserDataEnable", !"Negative"}
!301 = !{!"meshShaderWAPerPrimitiveUserDataEnablePatchName", !""}
!302 = !{!"generatePatchesForRTWriteSends", i1 false}
!303 = !{!"forceVMask", i1 false}
!304 = !{!"WaDisableVRS", i1 false}
!305 = !{!"RelaxMemoryVisibilityFromPSOrdering", i1 false}
!306 = !{!"WaEnableVMaskUnderNonUnifromCF", i1 false}
!307 = !{!"csInfo", !308, !309, !310, !311, !312, !20, !21, !313, !314, !315, !316, !317, !318, !319, !320, !321, !322, !323, !324, !325, !53, !326, !327, !328, !329, !330, !331, !332}
!308 = !{!"maxWorkGroupSize", i32 0}
!309 = !{!"waveSize", i32 0}
!310 = !{!"ComputeShaderSecondCompile"}
!311 = !{!"forcedSIMDSize", i8 0}
!312 = !{!"forceTotalGRFNum", i32 0}
!313 = !{!"forceSpillCompression", i1 false}
!314 = !{!"allowLowerSimd", i1 false}
!315 = !{!"disableSimd32Slicing", i1 false}
!316 = !{!"disableSplitOnSpill", i1 false}
!317 = !{!"enableNewSpillCostFunction", i1 false}
!318 = !{!"forceVISAPreSched", i1 false}
!319 = !{!"forceUniformBuffer", i1 false}
!320 = !{!"forceUniformSurfaceSampler", i1 false}
!321 = !{!"disableLocalIdOrderOptimizations", i1 false}
!322 = !{!"disableDispatchAlongY", i1 false}
!323 = !{!"neededThreadIdLayout", i1* null}
!324 = !{!"forceTileYWalk", i1 false}
!325 = !{!"atomicBranch", i32 0}
!326 = !{!"disableEarlyOut", i1 false}
!327 = !{!"walkOrderEnabled", i1 false}
!328 = !{!"walkOrderOverride", i32 0}
!329 = !{!"ResForHfPacking"}
!330 = !{!"hasWaveMatrix", i1 false}
!331 = !{!"constantFoldSimdSize", i1 false}
!332 = !{!"isNodeShader", i1 false}
!333 = !{!"msInfo", !334, !335, !336, !337, !338, !339, !340, !341, !342, !343, !344, !292, !290, !345}
!334 = !{!"PrimitiveTopology", i32 3}
!335 = !{!"MaxNumOfPrimitives", i32 0}
!336 = !{!"MaxNumOfVertices", i32 0}
!337 = !{!"MaxNumOfPerPrimitiveOutputs", i32 0}
!338 = !{!"MaxNumOfPerVertexOutputs", i32 0}
!339 = !{!"WorkGroupSize", i32 0}
!340 = !{!"WorkGroupMemorySizeInBytes", i32 0}
!341 = !{!"IndexFormat", i32 6}
!342 = !{!"SubgroupSize", i32 0}
!343 = !{!"VPandRTAIndexAutostripEnable", i1 false}
!344 = !{!"MeshShaderWAPerPrimitiveUserDataEnable", i1 false}
!345 = !{!"numPrimitiveAttributesPatchBaseName", !""}
!346 = !{!"taskInfo", !266, !339, !340, !342}
!347 = !{!"NBarrierCnt", i32 0}
!348 = !{!"rtInfo", !349, !350, !351, !352, !353, !354, !355, !356, !357, !358, !359, !360, !361, !362}
!349 = !{!"RayQueryAllocSizeInBytes", i32 0}
!350 = !{!"NumContinuations", i32 0}
!351 = !{!"RTAsyncStackAddrspace", i32 -1}
!352 = !{!"RTAsyncStackSurfaceStateOffset", i1* null}
!353 = !{!"SWHotZoneAddrspace", i32 -1}
!354 = !{!"SWHotZoneSurfaceStateOffset", i1* null}
!355 = !{!"SWStackAddrspace", i32 -1}
!356 = !{!"SWStackSurfaceStateOffset", i1* null}
!357 = !{!"RTSyncStackAddrspace", i32 -1}
!358 = !{!"RTSyncStackSurfaceStateOffset", i1* null}
!359 = !{!"doSyncDispatchRays", i1 false}
!360 = !{!"MemStyle", !"Xe"}
!361 = !{!"GlobalDataStyle", !"Xe"}
!362 = !{!"NeedsBTD", i1 true}
!363 = !{!"EnableTextureIndirection", i1 false}
!364 = !{!"EnableSamplerIndirection", i1 false}
!365 = !{!"samplerStateStride", i32 0}
!366 = !{!"samplerStateOffset", i32 0}
!367 = !{!"textureStateStride", i32 0}
!368 = !{!"textureStateOffset", i32 0}
!369 = !{!"CurUniqueIndirectIdx", i32 0}
!370 = !{!"inlineDynTextures"}
!371 = !{!"inlineResInfoData"}
!372 = !{!"immConstant", !373, !374, !375}
!373 = !{!"data"}
!374 = !{!"sizes"}
!375 = !{!"zeroIdxs"}
!376 = !{!"stringConstants"}
!377 = !{!"inlineBuffers", !378, !382, !383}
!378 = !{!"inlineBuffersVec[0]", !379, !380, !381}
!379 = !{!"alignment", i32 0}
!380 = !{!"allocSize", i64 0}
!381 = !{!"Buffer"}
!382 = !{!"inlineBuffersVec[1]", !379, !380, !381}
!383 = !{!"inlineBuffersVec[2]", !379, !380, !381}
!384 = !{!"GlobalPointerProgramBinaryInfos"}
!385 = !{!"ConstantPointerProgramBinaryInfos"}
!386 = !{!"GlobalBufferAddressRelocInfo"}
!387 = !{!"ConstantBufferAddressRelocInfo"}
!388 = !{!"forceLscCacheList"}
!389 = !{!"SrvMap"}
!390 = !{!"RootConstantBufferOffsetInBytes"}
!391 = !{!"RasterizerOrderedByteAddressBuffer"}
!392 = !{!"RasterizerOrderedViews"}
!393 = !{!"MinNOSPushConstantSize", i32 0}
!394 = !{!"inlineProgramScopeOffsets"}
!395 = !{!"shaderData", !396}
!396 = !{!"numReplicas", i32 0}
!397 = !{!"URBInfo", !398, !399, !400}
!398 = !{!"has64BVertexHeaderInput", i1 false}
!399 = !{!"has64BVertexHeaderOutput", i1 false}
!400 = !{!"hasVertexHeader", i1 true}
!401 = !{!"m_ForcePullModel", i1 false}
!402 = !{!"UseBindlessImage", i1 false}
!403 = !{!"enableRangeReduce", i1 false}
!404 = !{!"disableNewTrigFuncRangeReduction", i1 false}
!405 = !{!"enableFRemToSRemOpt", i1 false}
!406 = !{!"enableSampleptrToLdmsptrSample0", i1 false}
!407 = !{!"enableSampleLptrToLdmsptrSample0", i1 false}
!408 = !{!"WaForceSIMD32MicropolyRasterize", i1 false}
!409 = !{!"allowMatchMadOptimizationforVS", i1 false}
!410 = !{!"disableMatchMadOptimizationForCS", i1 false}
!411 = !{!"disableMemOptforNegativeOffsetLoads", i1 false}
!412 = !{!"enableThreeWayLoadSpiltOpt", i1 false}
!413 = !{!"statefulResourcesNotAliased", i1 false}
!414 = !{!"disableMixMode", i1 false}
!415 = !{!"genericAccessesResolved", i1 false}
!416 = !{!"disableSeparateSpillPvtScratchSpace", i1 false}
!417 = !{!"disableSeparateScratchWA", i1 false}
!418 = !{!"PrivateMemoryPerFG"}
!419 = !{!"m_OptsToDisable"}
!420 = !{!"capabilities", !421}
!421 = !{!"globalVariableDecorationsINTEL", i1 false}
!422 = !{!"m_ShaderResourceViewMcsMask", !423, !424}
!423 = !{!"m_ShaderResourceViewMcsMaskVec[0]", i64 0}
!424 = !{!"m_ShaderResourceViewMcsMaskVec[1]", i64 0}
!425 = !{!"computedDepthMode", i32 0}
!426 = !{!"isHDCFastClearShader", i1 false}
!427 = !{!"argRegisterReservations", !428}
!428 = !{!"argRegisterReservationsVec[0]", i32 0}
!429 = !{!"SIMD16_SpillThreshold", i8 0}
!430 = !{!"SIMD32_SpillThreshold", i8 0}
!431 = !{!"m_CacheControlOption", !432, !433, !434, !435}
!432 = !{!"LscLoadCacheControlOverride", i8 0}
!433 = !{!"LscStoreCacheControlOverride", i8 0}
!434 = !{!"TgmLoadCacheControlOverride", i8 0}
!435 = !{!"TgmStoreCacheControlOverride", i8 0}
!436 = !{i32 1, i32 0, i32 0, i32 0, i32 1, i32 0, i32 0, i32 1, i32 0, i32 0}
!437 = !{!"none", !"none", !"none", !"none", !"none", !"none", !"none", !"none", !"none", !"none"}
!438 = !{!"float*", !"class.sycl::_V1::range", !"class.sycl::_V1::range", !"long", !"class.sycl::_V1::ext::oneapi::bfloat16*", !"class.sycl::_V1::range", !"class.sycl::_V1::range", !"class.sycl::_V1::ext::oneapi::bfloat16*", !"class.sycl::_V1::range", !"class.sycl::_V1::range"}
!439 = !{!"", !"", !"", !"", !"", !"", !"", !"", !"", !""}
!440 = !{!441, !443, !443, !446, !447, !443, !443, !447, !443, !443}
!441 = !{!442}
!442 = !{i32 44, i32 4}
!443 = !{!444, !445}
!444 = !{i32 38, i32 2}
!445 = !{i32 44, i32 8}
!446 = !{}
!447 = !{!448}
!448 = !{i32 44, i32 2}
!449 = !{!450}
!450 = !{i32 4469}
!451 = !{!450, !452}
!452 = !{i32 4470}
!453 = !{!445}
