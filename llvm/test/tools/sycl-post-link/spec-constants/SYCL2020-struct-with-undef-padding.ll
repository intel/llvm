; RUN: sycl-post-link --spec-const=native -S < %s -o %t.files.table
; RUN: FileCheck %s -input-file=%t.files_0.ll --check-prefix CHECK-IR
; RUN: FileCheck %s -input-file=%t.files_0.prop --check-prefix CHECK-PROP
; RUN: %if asserts %{ sycl-post-link -debug-only=SpecConst --spec-const=native -S < %s 2>&1 | FileCheck %s --check-prefix=CHECK-LOG %}
;
; This test is intended to check that SpecConstantsPass is able to handle the
; situation where specialization constants with complex types such as structs
; have an 'undef' value for padding in LLVM IR

target datalayout = "e-i64:64-v16:16-v24:32-v32:32-v48:64-v96:128-v192:256-v256:256-v512:512-v1024:1024-n8:16:32:64"
target triple = "spir64-unknown-unknown"

%"class.cl::sycl::specialization_id" = type { %struct.coeff_str_aligned_t }
%"class.cl::sycl::specialization_id.1" = type { %struct.coeff2_str_aligned_t }
%struct.coeff_str_aligned_t = type { %"class.std::array", i64, [8 x i8] }
%struct.coeff2_str_aligned_t = type { %"class.std::array", i64, [7 x i8], i8 }
%"class.std::array" = type { [3 x float] }

$_ZTSZ4mainEUlN2cl4sycl14kernel_handlerEE_ = comdat any

@__usid_str = private unnamed_addr constant [32 x i8] c"ef880fa09cf7a9d7____ZL8coeff_id\00", align 1
@_ZL8coeff_id = internal addrspace(1) constant %"class.cl::sycl::specialization_id" { %struct.coeff_str_aligned_t { %"class.std::array" zeroinitializer, i64 0, [8 x i8] undef } }, align 32
@__usid_str.0 = private unnamed_addr constant [33 x i8] c"df991fa0adf9bad8____ZL8coeff_id2\00", align 1
@_ZL8coeff_id2 = internal addrspace(1) constant %"class.cl::sycl::specialization_id.1" { %struct.coeff2_str_aligned_t { %"class.std::array" zeroinitializer, i64 0, [7 x i8] undef, i8 undef } }, align 32

; Function Attrs: convergent norecurse
define weak_odr dso_local spir_kernel void @_ZTSZ4mainEUlN2cl4sycl14kernel_handlerEE_() local_unnamed_addr #0 comdat !kernel_arg_buffer_location !6 !sycl_kernel_omit_args !7 {
  %1 = alloca %struct.coeff_str_aligned_t, align 32
  %2 = addrspacecast %struct.coeff_str_aligned_t* %1 to %struct.coeff_str_aligned_t addrspace(4)*
  %3 = bitcast %struct.coeff_str_aligned_t* %1 to i8*
  call spir_func void @_Z40__sycl_getComposite2020SpecConstantValueI19coeff_str_aligned_tET_PKcPKvS5_(%struct.coeff_str_aligned_t addrspace(4)* sret(%struct.coeff_str_aligned_t) align 32 %2, i8 addrspace(4)* noundef addrspacecast (i8* getelementptr inbounds ([32 x i8], [32 x i8]* @__usid_str, i64 0, i64 0) to i8 addrspace(4)*), i8 addrspace(4)* noundef addrspacecast (i8 addrspace(1)* bitcast (%"class.cl::sycl::specialization_id" addrspace(1)* @_ZL8coeff_id to i8 addrspace(1)*) to i8 addrspace(4)*), i8 addrspace(4)* noundef null) #4
; CHECK-IR: %[[#NS0:]] = call float @_Z20__spirv_SpecConstantif(i32 [[#SCID0:]], float 0.000000e+00)
; CHECK-IR: %[[#NS1:]] = call float @_Z20__spirv_SpecConstantif(i32 [[#SCID1:]], float 0.000000e+00)
; CHECK-IR: %[[#NS2:]] = call float @_Z20__spirv_SpecConstantif(i32 [[#SCID2:]], float 0.000000e+00)
; CHECK-IR: %[[#NS3:]] = call [3 x float] @_Z29__spirv_SpecConstantCompositefff_RA3_f(float %[[#NS0]], float %[[#NS1]], float %[[#NS2]])
; CHECK-IR: %[[#NS4:]] = call %"class.std::array" @"_Z29__spirv_SpecConstantCompositeA3_f_Rclass.std::array"([3 x float] %[[#NS3]])
; CHECK-IR: %[[#NS5:]] = call i64 @_Z20__spirv_SpecConstantix(i32 [[#SCID3:]], i64 0)
; CHECK-IR: %[[#NS6:]] = call %struct.coeff_str_aligned_t @"_Z29__spirv_SpecConstantCompositeclass.std::arrayxA8_a_Rstruct.coeff_str_aligned_t"(%"class.std::array" %[[#NS4]], i64 %[[#NS5]], [8 x i8] undef)

  %4 = alloca %struct.coeff2_str_aligned_t, align 32
  %5 = addrspacecast %struct.coeff2_str_aligned_t* %4 to %struct.coeff2_str_aligned_t addrspace(4)*
  %6 = bitcast %struct.coeff2_str_aligned_t* %4 to i8*
  call spir_func void @_Z40__sycl_getComposite2020SpecConstantValueI19coeff2_str_aligned_tET_PKcPKvS5_(%struct.coeff2_str_aligned_t addrspace(4)* sret(%struct.coeff2_str_aligned_t) align 32 %5, i8 addrspace(4)* noundef addrspacecast (i8* getelementptr inbounds ([33 x i8], [33 x i8]* @__usid_str.0, i64 0, i64 0) to i8 addrspace(4)*), i8 addrspace(4)* noundef addrspacecast (i8 addrspace(1)* bitcast (%"class.cl::sycl::specialization_id.1" addrspace(1)* @_ZL8coeff_id2 to i8 addrspace(1)*) to i8 addrspace(4)*), i8 addrspace(4)* noundef null) #4
; CHECK-IR: %[[#NS7:]] = call float @_Z20__spirv_SpecConstantif(i32 [[#SCID4:]], float 0.000000e+00)
; CHECK-IR: %[[#NS8:]] = call float @_Z20__spirv_SpecConstantif(i32 [[#SCID5:]], float 0.000000e+00)
; CHECK-IR: %[[#NS9:]] = call float @_Z20__spirv_SpecConstantif(i32 [[#SCID6:]], float 0.000000e+00)
; CHECK-IR: %[[#NS10:]] = call [3 x float] @_Z29__spirv_SpecConstantCompositefff_RA3_f(float %[[#NS7]], float %[[#NS8]], float %[[#NS9]])
; CHECK-IR: %[[#NS11:]] = call %"class.std::array" @"_Z29__spirv_SpecConstantCompositeA3_f_Rclass.std::array"([3 x float] %[[#NS10]])
; CHECK-IR: %[[#NS12:]] = call i64 @_Z20__spirv_SpecConstantix(i32 [[#SCID7:]], i64 0)
; CHECK-IR: %[[#NS13:]] = call %struct.coeff2_str_aligned_t @"_Z29__spirv_SpecConstantCompositeclass.std::arrayxA7_aa_Rstruct.coeff2_str_aligned_t"(%"class.std::array" %[[#NS11]], i64 %[[#NS12]], [7 x i8] undef, i8 undef)

  ret void
}
; Function Attrs: convergent
declare dso_local spir_func void @_Z40__sycl_getComposite2020SpecConstantValueI19coeff_str_aligned_tET_PKcPKvS5_(%struct.coeff_str_aligned_t addrspace(4)* sret(%struct.coeff_str_aligned_t) align 32, i8 addrspace(4)* noundef, i8 addrspace(4)* noundef, i8 addrspace(4)* noundef) local_unnamed_addr #2

declare dso_local spir_func void @_Z40__sycl_getComposite2020SpecConstantValueI19coeff2_str_aligned_tET_PKcPKvS5_(%struct.coeff2_str_aligned_t addrspace(4)* sret(%struct.coeff2_str_aligned_t) align 32, i8 addrspace(4)* noundef, i8 addrspace(4)* noundef, i8 addrspace(4)* noundef) local_unnamed_addr #2

attributes #0 = { convergent norecurse "frame-pointer"="all" "min-legal-vector-width"="0" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "sycl-module-id"="spec-constant-test.cpp" "uniform-work-group-size"="true" }
attributes #1 = { argmemonly mustprogress nofree nosync nounwind willreturn }
attributes #2 = { convergent "frame-pointer"="all" "no-trapping-math"="true" "stack-protector-buffer-size"="8" }
attributes #3 = { nounwind }
attributes #4 = { convergent }

!llvm.dependent-libraries = !{!0}
!llvm.module.flags = !{!1, !2}
!opencl.spir.version = !{!3}
!spirv.Source = !{!4}
!llvm.ident = !{!5}
; CHECK-IR: !sycl.specialization-constants = !{![[#MN0:]], ![[#MN1:]]}
; CHECK-IR: !sycl.specialization-constants-default-values = !{![[#MN2:]], ![[#MN3:]]}

!0 = !{!"libcpmt"}
!1 = !{i32 1, !"wchar_size", i32 2}
!2 = !{i32 7, !"frame-pointer", i32 2}
!3 = !{i32 1, i32 2}
!4 = !{i32 4, i32 100000}
!5 = !{!"clang version 14.0.0"}
!6 = !{i32 -1}
!7 = !{i1 true}
; CHECK-IR: ![[#MN0]] = !{!"ef880fa09cf7a9d7____ZL8coeff_id", i32 0, i32 0, i32 4, i32 1, i32 4, i32 4, i32 2, i32 8, i32 4, i32 3, i32 16, i32 8, i32 -1, i32 24, i32 8}
; CHECK-IR: ![[#MN1]] = !{!"df991fa0adf9bad8____ZL8coeff_id2", i32 5, i32 0, i32 4, i32 6, i32 4, i32 4, i32 7, i32 8, i32 4, i32 8, i32 16, i32 8, i32 -1, i32 31, i32 1}
; CHECK-IR: ![[#MN2]] = !{%struct.coeff_str_aligned_t { %"class.std::array" zeroinitializer, i64 0, [8 x i8] undef }}
; CHECK-IR: ![[#MN3]] = !{%struct.coeff2_str_aligned_t { %"class.std::array" zeroinitializer, i64 0, [7 x i8] undef, i8 undef }}

; CHECK-PROP: [SYCL/specialization constants]
; CHECK-PROP-NEXT: ef880fa09cf7a9d7____ZL8coeff_id=2|
; CHECK-PROP-NEXT: df991fa0adf9bad8____ZL8coeff_id2=2|
; CHECK-LOG: sycl.specialization-constants
; CHECK-LOG:[[UNIQUE_PREFIX:[0-9a-zA-Z]+]]={0, 0, 4}
; CHECK-LOG:[[UNIQUE_PREFIX]]={1, 4, 4}
; CHECK-LOG:[[UNIQUE_PREFIX]]={2, 8, 4}
; CHECK-LOG:[[UNIQUE_PREFIX]]={3, 16, 8}
; CHECK-LOG:[[UNIQUE_PREFIX]]={4294967295, 24, 8}
; CHECK-LOG:[[UNIQUE_PREFIX2:[0-9a-zA-Z]+]]={5, 0, 4}
; CHECK-LOG:[[UNIQUE_PREFIX2]]={6, 4, 4}
; CHECK-LOG:[[UNIQUE_PREFIX2]]={7, 8, 4}
; CHECK-LOG:[[UNIQUE_PREFIX2]]={8, 16, 8}
; CHECK-LOG:[[UNIQUE_PREFIX2]]={4294967295, 31, 1}

; CHECK-PROP: [SYCL/specialization constants default values]
; CHECK-PROP-NEXT: all=2|
; CHECK-LOG: sycl.specialization-constants-default-values
; CHECK-LOG: {0, 12, 0}
; CHECK-LOG: {16, 8, 0}
; CHECK-LOG: {24, 8, 0}
; CHECK-LOG: {32, 12, 0}
; CHECK-LOG: {48, 8, 0}
; CHECK-LOG: {56, 7, 0}
; CHECK-LOG: {63, 1, 0}
