; RUN: sycl-post-link --ir-output-only --spec-const=rt %s -S -o - | FileCheck %s
;
; This test is intended to check that SpecConstantsPass is able to handle the
; situation where specialization constants with complex types such as arrays
; within arrays have zeroinitializer in LLVM IR

target datalayout = "e-i64:64-v16:16-v24:32-v32:32-v48:64-v96:128-v192:256-v256:256-v512:512-v1024:1024-n8:16:32:64"
target triple = "spir64-unknown-unknown"

%"class.cl::sycl::specialization_id" = type { %"class.std::array" }
%"class.std::array" = type { [3 x %"class.std::array.1"] }
%"class.std::array.1" = type { [3 x float] }
%"class.cl::sycl::kernel_handler" = type { i8 addrspace(4)* }

@__usid_str = private unnamed_addr constant [32 x i8] c"9f47062a80eecfa7____ZL8coeff_id\00", align 1
@_ZL8coeff_id = internal addrspace(1) constant %"class.cl::sycl::specialization_id" zeroinitializer, align 4

; Function Attrs: convergent mustprogress norecurse
define internal spir_func void @_ZN2cl4sycl14kernel_handler33getSpecializationConstantOnDeviceIL_ZL8coeff_idESt5arrayIS3_IfLy3EELy3EELPv0EEET0_v(%"class.std::array" addrspace(4)* noalias sret(%"class.std::array") align 4 %0, %"class.cl::sycl::kernel_handler" addrspace(4)* align 8 dereferenceable_or_null(8) %1) #0 align 2 {
  %3 = alloca %"class.cl::sycl::kernel_handler" addrspace(4)*, align 8
  %4 = alloca i8 addrspace(4)*, align 8
  %5 = alloca i32, align 4
  %6 = addrspacecast %"class.cl::sycl::kernel_handler" addrspace(4)** %3 to %"class.cl::sycl::kernel_handler" addrspace(4)* addrspace(4)*
  %7 = addrspacecast i8 addrspace(4)** %4 to i8 addrspace(4)* addrspace(4)*
  store %"class.cl::sycl::kernel_handler" addrspace(4)* %1, %"class.cl::sycl::kernel_handler" addrspace(4)* addrspace(4)* %6, align 8, !tbaa !4
  %8 = load %"class.cl::sycl::kernel_handler" addrspace(4)*, %"class.cl::sycl::kernel_handler" addrspace(4)* addrspace(4)* %6, align 8
  %9 = bitcast i8 addrspace(4)** %4 to i8*
  store i8 addrspace(4)* addrspacecast (i8* getelementptr inbounds ([32 x i8], [32 x i8]* @__usid_str, i32 0, i32 0) to i8 addrspace(4)*), i8 addrspace(4)* addrspace(4)* %7, align 8, !tbaa !4
  %10 = load i8 addrspace(4)*, i8 addrspace(4)* addrspace(4)* %7, align 8, !tbaa !4
  %11 = getelementptr inbounds %"class.cl::sycl::kernel_handler", %"class.cl::sycl::kernel_handler" addrspace(4)* %8, i32 0, i32 0
  %12 = load i8 addrspace(4)*, i8 addrspace(4)* addrspace(4)* %11, align 8, !tbaa !8
  call spir_func void @_Z40__sycl_getComposite2020SpecConstantValueISt5arrayIS0_IfLy3EELy3EEET_PKcPKvS7_(%"class.std::array" addrspace(4)* sret(%"class.std::array") align 4 %0, i8 addrspace(4)* %10, i8 addrspace(4)* addrspacecast (i8 addrspace(1)* bitcast (%"class.cl::sycl::specialization_id" addrspace(1)* @_ZL8coeff_id to i8 addrspace(1)*) to i8 addrspace(4)*), i8 addrspace(4)* %12) #13
; CHECK: %[[#NS0:]] = call float @_Z20__spirv_SpecConstantif(i32 [[#SCID0:]], float 0.000000e+00)
; CHECK: %[[#NS1:]] = call float @_Z20__spirv_SpecConstantif(i32 [[#SCID1:]], float 0.000000e+00)
; CHECK: %[[#NS2:]] = call float @_Z20__spirv_SpecConstantif(i32 [[#SCID2:]], float 0.000000e+00)
; CHECK: %[[#NS3:]] = call [3 x float] @_Z29__spirv_SpecConstantCompositefff_RA3_f(float %[[#NS0]], float %[[#NS1]], float %[[#NS2]])
; CHECK: %[[#NS4:]] = call %"class.std::array.1" @"_Z29__spirv_SpecConstantCompositeA3_f_Rclass.std::array.1"([3 x float] %[[#NS3]])
; CHECK: %[[#NS5:]] = call float @_Z20__spirv_SpecConstantif(i32 [[#SCID3:]], float 0.000000e+00)
; CHECK: %[[#NS6:]] = call float @_Z20__spirv_SpecConstantif(i32 [[#SCID4:]], float 0.000000e+00)
; CHECK: %[[#NS7:]] = call float @_Z20__spirv_SpecConstantif(i32 [[#SCID5:]], float 0.000000e+00)
; CHECK: %[[#NS8:]] = call [3 x float] @_Z29__spirv_SpecConstantCompositefff_RA3_f(float %[[#NS5]], float %[[#NS6]], float %[[#NS7]])
; CHECK: %[[#NS9:]] = call %"class.std::array.1" @"_Z29__spirv_SpecConstantCompositeA3_f_Rclass.std::array.1"([3 x float] %[[#NS8]])
; CHECK: %[[#NS10:]] = call float @_Z20__spirv_SpecConstantif(i32 [[#SCID6:]], float 0.000000e+00)
; CHECK: %[[#NS11:]] = call float @_Z20__spirv_SpecConstantif(i32 [[#SCID7:]], float 0.000000e+00)
; CHECK: %[[#NS12:]] = call float @_Z20__spirv_SpecConstantif(i32 [[#SCID8:]], float 0.000000e+00)
; CHECK: %[[#NS13:]] = call [3 x float] @_Z29__spirv_SpecConstantCompositefff_RA3_f(float %[[#NS10]], float %[[#NS11]], float %[[#NS12]])
; CHECK: %[[#NS14:]] = call %"class.std::array.1" @"_Z29__spirv_SpecConstantCompositeA3_f_Rclass.std::array.1"([3 x float] %[[#NS13]])
; CHECK: %[[#NS15:]] = call [3 x %"class.std::array.1"] @"_Z29__spirv_SpecConstantCompositeclass.std::array.1class.std::array.1class.std::array.1_RA3_class.std::array.1"(%"class.std::array.1" %[[#NS4]], %"class.std::array.1" %[[#NS9]], %"class.std::array.1" %[[#NS14]])
; CHECK: %[[#NS16:]] = call %"class.std::array" @"_Z29__spirv_SpecConstantCompositeA3_class.std::array.1_Rclass.std::array"([3 x %"class.std::array.1"] %[[#NS15]])

  %13 = bitcast i8 addrspace(4)** %4 to i8*
  ret void
}

; Function Attrs: convergent
declare dso_local spir_func void @_Z40__sycl_getComposite2020SpecConstantValueISt5arrayIS0_IfLy3EELy3EEET_PKcPKvS7_(%"class.std::array" addrspace(4)* sret(%"class.std::array") align 4, i8 addrspace(4)*, i8 addrspace(4)*, i8 addrspace(4)*) #1

attributes #0 = { convergent mustprogress norecurse "frame-pointer"="all" "min-legal-vector-width"="0" "no-trapping-math"="true" "stack-protector-buffer-size"="8" }
attributes #1 = { convergent "frame-pointer"="all" "no-trapping-math"="true" "stack-protector-buffer-size"="8" }

!opencl.spir.version = !{!1, !1, !1, !1, !1, !1, !1, !1, !1, !1, !1, !1}
!spirv.Source = !{!2, !2, !2, !2, !2, !2, !2, !2, !2, !2, !2, !2}
!llvm.ident = !{!3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3}

!1 = !{i32 1, i32 2}
!2 = !{i32 4, i32 100000}
!3 = !{!"clang version 14.0.0"}
!4 = !{!5, !5, i64 0}
!5 = !{!"any pointer", !6, i64 0}
!6 = !{!"omnipotent char", !7, i64 0}
!7 = !{!"Simple C++ TBAA"}
!8 = !{!9, !5, i64 0}
!9 = !{!"_ZTSN2cl4sycl14kernel_handlerE", !5, i64 0}
