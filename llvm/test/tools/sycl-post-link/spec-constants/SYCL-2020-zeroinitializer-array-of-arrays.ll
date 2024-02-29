; RUN: sycl-post-link --spec-const=native -S < %s -o %t.files.table
; RUN: FileCheck %s -input-file=%t.files_0.ll --check-prefix CHECK-IR
; RUN: FileCheck %s -input-file=%t.files_0.prop --check-prefix CHECK-PROP
; RUN: %if asserts %{ sycl-post-link -debug-only=SpecConst --spec-const=native -S < %s 2>&1 | FileCheck %s --check-prefix=CHECK-LOG %}
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
%"class.cl::sycl::specialization_id.1" = type { %struct.coeff_str_t }
%struct.coeff_str_t = type { %"class.std::array.1", i64 }

@__usid_str.1 = private unnamed_addr constant [32 x i8] c"9f47062a80eecfa7____ZL8coeff_id\00", align 1
@_ZL8coeff_id = weak_odr addrspace(1) constant %"class.cl::sycl::specialization_id" zeroinitializer, align 4

@__usid_str.2 = private unnamed_addr constant [33 x i8] c"405761736d5a1797____ZL9coeff_id2\00", align 1
@_ZL9coeff_id2 = weak_odr addrspace(1) constant %"class.cl::sycl::specialization_id" { %"class.std::array" { [3 x %"class.std::array.1"] [%"class.std::array.1" zeroinitializer, %"class.std::array.1" { [3 x float] [float 0.000000e+00, float 1.000000e+00, float 2.000000e+00] }, %"class.std::array.1" { [3 x float] [float 0x4010666660000000, float 0x4014666660000000, float 0x4018CCCCC0000000] }] } }, align 4

@__usid_str.3 = private unnamed_addr constant [33 x i8] c"6da74a122db9f35d____ZL9coeff_id3\00", align 1
@_ZL9coeff_id3 = weak_odr addrspace(1) constant %"class.cl::sycl::specialization_id.1" zeroinitializer, align 8

; Function Attrs: convergent mustprogress norecurse
define weak_odr spir_func void @_ZN2cl4sycl14kernel_handler33getSpecializationConstantOnDeviceIL_ZL8coeff_idESt5arrayIS3_IfLy3EELy3EELPv0EEET0_v(%"class.std::array" addrspace(4)* noalias sret(%"class.std::array") align 4 %0, %"class.cl::sycl::kernel_handler" addrspace(4)* align 8 dereferenceable_or_null(8) %1) #0 align 2 {
  %3 = alloca %"class.cl::sycl::kernel_handler" addrspace(4)*, align 8
  %4 = alloca i8 addrspace(4)*, align 8
  %5 = addrspacecast %"class.cl::sycl::kernel_handler" addrspace(4)** %3 to %"class.cl::sycl::kernel_handler" addrspace(4)* addrspace(4)*
  %6 = addrspacecast i8 addrspace(4)** %4 to i8 addrspace(4)* addrspace(4)*
  store %"class.cl::sycl::kernel_handler" addrspace(4)* %1, %"class.cl::sycl::kernel_handler" addrspace(4)* addrspace(4)* %5, align 8, !tbaa !4
  %7 = load %"class.cl::sycl::kernel_handler" addrspace(4)*, %"class.cl::sycl::kernel_handler" addrspace(4)* addrspace(4)* %5, align 8
  store i8 addrspace(4)* addrspacecast (i8* getelementptr inbounds ([32 x i8], [32 x i8]* @__usid_str.1, i32 0, i32 0) to i8 addrspace(4)*), i8 addrspace(4)* addrspace(4)* %6, align 8, !tbaa !4
  %8 = load i8 addrspace(4)*, i8 addrspace(4)* addrspace(4)* %6, align 8, !tbaa !4
  %9 = getelementptr inbounds %"class.cl::sycl::kernel_handler", %"class.cl::sycl::kernel_handler" addrspace(4)* %7, i32 0, i32 0
  %10 = load i8 addrspace(4)*, i8 addrspace(4)* addrspace(4)* %9, align 8, !tbaa !8
  call spir_func void @_Z40__sycl_getComposite2020SpecConstantValueISt5arrayIS0_IfLy3EELy3EEET_PKcPKvS7_(%"class.std::array" addrspace(4)* sret(%"class.std::array") align 4 %0, i8 addrspace(4)* %8, i8 addrspace(4)* addrspacecast (i8 addrspace(1)* bitcast (%"class.cl::sycl::specialization_id" addrspace(1)* @_ZL8coeff_id to i8 addrspace(1)*) to i8 addrspace(4)*), i8 addrspace(4)* %10) #13
; CHECK-IR: %[[#NS0:]] = call float @_Z20__spirv_SpecConstantif(i32 [[#SCID0:]], float 0.000000e+00)
; CHECK-IR: %[[#NS1:]] = call float @_Z20__spirv_SpecConstantif(i32 [[#SCID1:]], float 0.000000e+00)
; CHECK-IR: %[[#NS2:]] = call float @_Z20__spirv_SpecConstantif(i32 [[#SCID2:]], float 0.000000e+00)
; CHECK-IR: %[[#NS3:]] = call [3 x float] @_Z29__spirv_SpecConstantCompositefff_RA3_f(float %[[#NS0]], float %[[#NS1]], float %[[#NS2]])
; CHECK-IR: %[[#NS4:]] = call %"class.std::array.1" @"_Z29__spirv_SpecConstantCompositeA3_f_Rclass.std::array.1"([3 x float] %[[#NS3]])
; CHECK-IR: %[[#NS5:]] = call float @_Z20__spirv_SpecConstantif(i32 [[#SCID3:]], float 0.000000e+00)
; CHECK-IR: %[[#NS6:]] = call float @_Z20__spirv_SpecConstantif(i32 [[#SCID4:]], float 0.000000e+00)
; CHECK-IR: %[[#NS7:]] = call float @_Z20__spirv_SpecConstantif(i32 [[#SCID5:]], float 0.000000e+00)
; CHECK-IR: %[[#NS8:]] = call [3 x float] @_Z29__spirv_SpecConstantCompositefff_RA3_f(float %[[#NS5]], float %[[#NS6]], float %[[#NS7]])
; CHECK-IR: %[[#NS9:]] = call %"class.std::array.1" @"_Z29__spirv_SpecConstantCompositeA3_f_Rclass.std::array.1"([3 x float] %[[#NS8]])
; CHECK-IR: %[[#NS10:]] = call float @_Z20__spirv_SpecConstantif(i32 [[#SCID6:]], float 0.000000e+00)
; CHECK-IR: %[[#NS11:]] = call float @_Z20__spirv_SpecConstantif(i32 [[#SCID7:]], float 0.000000e+00)
; CHECK-IR: %[[#NS12:]] = call float @_Z20__spirv_SpecConstantif(i32 [[#SCID8:]], float 0.000000e+00)
; CHECK-IR: %[[#NS13:]] = call [3 x float] @_Z29__spirv_SpecConstantCompositefff_RA3_f(float %[[#NS10]], float %[[#NS11]], float %[[#NS12]])
; CHECK-IR: %[[#NS14:]] = call %"class.std::array.1" @"_Z29__spirv_SpecConstantCompositeA3_f_Rclass.std::array.1"([3 x float] %[[#NS13]])
; CHECK-IR: %[[#NS15:]] = call [3 x %"class.std::array.1"] @"_Z29__spirv_SpecConstantCompositeclass.std::array.1class.std::array.1class.std::array.1_RA3_class.std::array.1"(%"class.std::array.1" %[[#NS4]], %"class.std::array.1" %[[#NS9]], %"class.std::array.1" %[[#NS14]])
; CHECK-IR: %[[#NS16:]] = call %"class.std::array" @"_Z29__spirv_SpecConstantCompositeA3_class.std::array.1_Rclass.std::array"([3 x %"class.std::array.1"] %[[#NS15]])

  %11 = alloca %"class.std::array", align 4
  %12 = addrspacecast %"class.std::array"* %11 to %"class.std::array" addrspace(4)*
  call spir_func void @_Z40__sycl_getComposite2020SpecConstantValueISt5arrayIS0_IfLy3EELy3EEET_PKcPKvS7_(%"class.std::array" addrspace(4)* sret(%"class.std::array") align 4 %12, i8 addrspace(4)* addrspacecast (i8* getelementptr inbounds ([33 x i8], [33 x i8]* @__usid_str.2, i64 0, i64 0) to i8 addrspace(4)*), i8 addrspace(4)* addrspacecast (i8 addrspace(1)* bitcast (%"class.cl::sycl::specialization_id" addrspace(1)* @_ZL9coeff_id2 to i8 addrspace(1)*) to i8 addrspace(4)*), i8 addrspace(4)* null) #13
; CHECK-IR: %[[#NS17:]] = call float @_Z20__spirv_SpecConstantif(i32 [[#SCID9:]], float 0.000000e+00)
; CHECK-IR: %[[#NS18:]] = call float @_Z20__spirv_SpecConstantif(i32 [[#SCID10:]], float 0.000000e+00)
; CHECK-IR: %[[#NS19:]] = call float @_Z20__spirv_SpecConstantif(i32 [[#SCID11:]], float 0.000000e+00)
; CHECK-IR: %[[#NS20:]] = call [3 x float] @_Z29__spirv_SpecConstantCompositefff_RA3_f(float %[[#NS17]], float %[[#NS18]], float %[[#NS19]])
; CHECK-IR: %[[#NS21:]] = call %"class.std::array.1" @"_Z29__spirv_SpecConstantCompositeA3_f_Rclass.std::array.1"([3 x float] %[[#NS20]])
; CHECK-IR: %[[#NS22:]] = call float @_Z20__spirv_SpecConstantif(i32 [[#SCID12:]], float 0.000000e+00)
; CHECK-IR: %[[#NS23:]] = call float @_Z20__spirv_SpecConstantif(i32 [[#SCID13:]], float 1.000000e+00)
; CHECK-IR: %[[#NS24:]] = call float @_Z20__spirv_SpecConstantif(i32 [[#SCID14:]], float 2.000000e+00)
; CHECK-IR: %[[#NS25:]] = call [3 x float] @_Z29__spirv_SpecConstantCompositefff_RA3_f(float %[[#NS22]], float %[[#NS23]], float %[[#NS24]])
; CHECK-IR: %[[#NS26:]] = call %"class.std::array.1" @"_Z29__spirv_SpecConstantCompositeA3_f_Rclass.std::array.1"([3 x float] %[[#NS25]])
; CHECK-IR: %[[#NS27:]] = call float @_Z20__spirv_SpecConstantif(i32 [[#SCID15:]], float 0x4010666660000000)
; CHECK-IR: %[[#NS28:]] = call float @_Z20__spirv_SpecConstantif(i32 [[#SCID16:]], float 0x4014666660000000)
; CHECK-IR: %[[#NS29:]] = call float @_Z20__spirv_SpecConstantif(i32 [[#SCID17:]], float 0x4018CCCCC0000000)
; CHECK-IR: %[[#NS30:]] = call [3 x float] @_Z29__spirv_SpecConstantCompositefff_RA3_f(float %[[#NS27]], float %[[#NS28]], float %[[#NS29]])
; CHECK-IR: %[[#NS31:]] = call %"class.std::array.1" @"_Z29__spirv_SpecConstantCompositeA3_f_Rclass.std::array.1"([3 x float] %[[#NS30]])
; CHECK-IR: %[[#NS32:]] = call [3 x %"class.std::array.1"] @"_Z29__spirv_SpecConstantCompositeclass.std::array.1class.std::array.1class.std::array.1_RA3_class.std::array.1"(%"class.std::array.1" %[[#NS21]], %"class.std::array.1" %[[#NS26]], %"class.std::array.1" %[[#NS31]])
; CHECK-IR: %[[#NS33:]] = call %"class.std::array" @"_Z29__spirv_SpecConstantCompositeA3_class.std::array.1_Rclass.std::array"([3 x %"class.std::array.1"] %[[#NS32]])
  
  %13 = alloca %struct.coeff_str_t, align 8
  %14 = addrspacecast %struct.coeff_str_t* %13 to %struct.coeff_str_t addrspace(4)*
  call spir_func void @_Z40__sycl_getComposite2020SpecConstantValueI11coeff_str_tET_PKcPKvS5_(%struct.coeff_str_t addrspace(4)* sret(%struct.coeff_str_t) align 8 %14, i8 addrspace(4)* addrspacecast (i8* getelementptr inbounds ([33 x i8], [33 x i8]* @__usid_str.3, i64 0, i64 0) to i8 addrspace(4)*), i8 addrspace(4)* addrspacecast (i8 addrspace(1)* bitcast (%"class.cl::sycl::specialization_id.1" addrspace(1)* @_ZL9coeff_id3 to i8 addrspace(1)*) to i8 addrspace(4)*), i8 addrspace(4)* null) #13
  ret void
}

; Function Attrs: convergent
declare dso_local spir_func void @_Z40__sycl_getComposite2020SpecConstantValueISt5arrayIS0_IfLy3EELy3EEET_PKcPKvS7_(%"class.std::array" addrspace(4)* sret(%"class.std::array") align 4, i8 addrspace(4)*, i8 addrspace(4)*, i8 addrspace(4)*) #1

; Function Attrs: convergent
declare dso_local spir_func void @_Z40__sycl_getComposite2020SpecConstantValueI11coeff_str_tET_PKcPKvS5_(%struct.coeff_str_t addrspace(4)* sret(%struct.coeff_str_t) align 8, i8 addrspace(4)*, i8 addrspace(4)*, i8 addrspace(4)*) local_unnamed_addr #2

attributes #0 = { convergent mustprogress norecurse "frame-pointer"="all" "min-legal-vector-width"="0" "no-trapping-math"="true" "stack-protector-buffer-size"="8" }
attributes #1 = { convergent "frame-pointer"="all" "no-trapping-math"="true" "stack-protector-buffer-size"="8" }
attributes #2 = { convergent "frame-pointer"="all" "no-trapping-math"="true" "stack-protector-buffer-size"="8" }

!opencl.spir.version = !{!1}
!spirv.Source = !{!2}
!llvm.ident = !{!3}

!1 = !{i32 1, i32 2}
!2 = !{i32 4, i32 100000}
!3 = !{!"clang version 14.0.0"}
!4 = !{!5, !5, i64 0}
!5 = !{!"any pointer", !6, i64 0}
!6 = !{!"omnipotent char", !7, i64 0}
!7 = !{!"Simple C++ TBAA"}
!8 = !{!9, !5, i64 0}
!9 = !{!"_ZTSN2cl4sycl14kernel_handlerE", !5, i64 0}

; CHECK-PROP: [SYCL/specialization constants]
; CHECK-PROP-NEXT: 9f47062a80eecfa7____ZL8coeff_id=2
; CHECK-PROP-NEXT: 405761736d5a1797____ZL9coeff_id2=2
; CHECK-PROP-NEXT: 6da74a122db9f35d____ZL9coeff_id3=2
; CHECK-LOG: sycl.specialization-constants
; CHECK-LOG:[[UNIQUE_PREFIX:[0-9a-zA-Z]+]]={0, 0, 4}
; CHECK-LOG:[[UNIQUE_PREFIX]]={1, 4, 4}
; CHECK-LOG:[[UNIQUE_PREFIX]]={2, 8, 4}
; CHECK-LOG:[[UNIQUE_PREFIX]]={3, 12, 4}
; CHECK-LOG:[[UNIQUE_PREFIX]]={4, 16, 4}
; CHECK-LOG:[[UNIQUE_PREFIX]]={5, 20, 4}
; CHECK-LOG:[[UNIQUE_PREFIX]]={6, 24, 4}
; CHECK-LOG:[[UNIQUE_PREFIX]]={7, 28, 4}
; CHECK-LOG:[[UNIQUE_PREFIX]]={8, 32, 4}
; CHECK-LOG:[[UNIQUE_PREFIX2:[0-9a-zA-Z]+]]={9, 0, 4}
; CHECK-LOG:[[UNIQUE_PREFIX2]]={10, 4, 4}
; CHECK-LOG:[[UNIQUE_PREFIX2]]={11, 8, 4}
; CHECK-LOG:[[UNIQUE_PREFIX2]]={12, 12, 4}
; CHECK-LOG:[[UNIQUE_PREFIX2]]={13, 16, 4}
; CHECK-LOG:[[UNIQUE_PREFIX2]]={14, 20, 4}
; CHECK-LOG:[[UNIQUE_PREFIX2]]={15, 24, 4}
; CHECK-LOG:[[UNIQUE_PREFIX2]]={16, 28, 4}
; CHECK-LOG:[[UNIQUE_PREFIX2]]={17, 32, 4}
; CHECK-LOG:[[UNIQUE_PREFIX3:[0-9a-zA-Z]+]]={18, 0, 4}
; CHECK-LOG:[[UNIQUE_PREFIX3]]={19, 4, 4}
; CHECK-LOG:[[UNIQUE_PREFIX3]]={20, 8, 4}
; CHECK-LOG:[[UNIQUE_PREFIX3]]={21, 16, 8}


; CHECK-PROP: [SYCL/specialization constants default values]
; CHECK-PROP-NEXT: all=2
; CHECK-LOG: sycl.specialization-constants-default-values
; CHECK-LOG: {0, 36, 0}
; CHECK-LOG: {36, 12, 0}
; CHECK-LOG: {48, 4, 0.000000e+00}
; CHECK-LOG: {52, 4, 1.000000e+00}
; CHECK-LOG: {56, 4, 2.000000e+00}
; CHECK-LOG: {60, 4, 4.100000e+00}
; CHECK-LOG: {64, 4, 5.100000e+00}
; CHECK-LOG: {68, 4, 6.200000e+00}
; CHECK-LOG: {72, 24, 0}
