; RUN: sycl-post-link --spec-const=rt -S %s -o %t.files.table
; RUN: FileCheck %s -input-file=%t.files_0.ll --check-prefix CHECK-IR
; RUN: FileCheck %s -input-file=%t.files_0.prop --check-prefix CHECK-PROP
;
; This test is intended to check that SpecConstantsPass is able to handle the
; situation where specialization constants with complex types such as structs
; have an 'undef' value for padding in LLVM IR

target datalayout = "e-i64:64-v16:16-v24:32-v32:32-v48:64-v96:128-v192:256-v256:256-v512:512-v1024:1024-n8:16:32:64"
target triple = "spir64-unknown-unknown"

%"class.cl::sycl::specialization_id" = type { %struct.coeff_str_aligned_t }
%struct.coeff_str_aligned_t = type { %"class.std::array", i64, [8 x i8] }
%"class.std::array" = type { [3 x float] }

$_ZTSZ4mainEUlN2cl4sycl14kernel_handlerEE_ = comdat any

@__usid_str = private unnamed_addr constant [32 x i8] c"ef880fa09cf7a9d7____ZL8coeff_id\00", align 1
@_ZL8coeff_id = internal addrspace(1) constant %"class.cl::sycl::specialization_id" { %struct.coeff_str_aligned_t { %"class.std::array" zeroinitializer, i64 0, [8 x i8] undef } }, align 32

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

  ret void
}
; Function Attrs: convergent
declare dso_local spir_func void @_Z40__sycl_getComposite2020SpecConstantValueI19coeff_str_aligned_tET_PKcPKvS5_(%struct.coeff_str_aligned_t addrspace(4)* sret(%struct.coeff_str_aligned_t) align 32, i8 addrspace(4)* noundef, i8 addrspace(4)* noundef, i8 addrspace(4)* noundef) local_unnamed_addr #2

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

!0 = !{!"libcpmt"}
!1 = !{i32 1, !"wchar_size", i32 2}
!2 = !{i32 7, !"frame-pointer", i32 2}
!3 = !{i32 1, i32 2}
!4 = !{i32 4, i32 100000}
!5 = !{!"clang version 14.0.0"}
!6 = !{i32 -1}
!7 = !{i1 true}

; CHECK-PROP: [SYCL/specialization constants]
; CHECK-PROP-NEXT: ef880fa09cf7a9d7____ZL8coeff_id=2|

; CHECK-PROP: [SYCL/specialization constants default values]
; CHECK-PROP-NEXT: all=2|
