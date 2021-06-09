; RUN: sycl-post-link -spec-const=rt --ir-output-only %s -S -o - \
; RUN: | FileCheck %s --implicit-check-not "call {{.*}} __sycl_getCompositeSpecConstantValue" --implicit-check-not "call {{.*}} __sycl_getComposite2020SpecConstantValue"
;
; This test is intended to check that sycl-post-link tool is capable of handling
; composite specialization constants by lowering them into a set of SPIR-V
; friendly IR operations representing those constants.
;
; CHECK-LABEL: define {{.*}} spir_kernel void @_ZTS4Test
; CHECK: %[[#NS0:]] = call i32 @_Z20__spirv_SpecConstantii(i32 [[#ID:]], i32
; CHECK: %[[#NS1:]] = call float @_Z20__spirv_SpecConstantif(i32 [[#ID + 1]], float
; CHECK: %[[#NA0:]] = call %struct._ZTS1A.A @_Z29__spirv_SpecConstantCompositeif(i32 %[[#NS0]], float %[[#NS1]])
;
; CHECK: %[[#NS2:]] = call i32 @_Z20__spirv_SpecConstantii(i32 [[#ID + 2]], i32
; CHECK: %[[#NS3:]] = call float @_Z20__spirv_SpecConstantif(i32 [[#ID + 3]], float
; CHECK: %[[#NA1:]] = call %struct._ZTS1A.A @_Z29__spirv_SpecConstantCompositeif(i32 %[[#NS2]], float %[[#NS3]])
;
; CHECK: %[[#NA:]] = call [2 x %struct._ZTS1A.A] @_Z29__spirv_SpecConstantCompositestruct._ZTS1A.Astruct._ZTS1A.A(%struct._ZTS1A.A %[[#NA0]], %struct._ZTS1A.A %[[#NA1]])
;
; CHECK: %[[#B0:]] = call i32 @_Z20__spirv_SpecConstantii(i32 [[#ID + 4]], i32{{.*}})
; CHECK: %[[#B1:]] = call i32 @_Z20__spirv_SpecConstantii(i32 [[#ID + 5]], i32{{.*}})
; CHECK: %[[#BV:]] = call <2 x i32> @_Z29__spirv_SpecConstantCompositeii(i32 %[[#B0]], i32 %[[#B1]])
; CHECK: %[[#B:]] = call %"class._ZTSN2cl4sycl3vecIiLi2EEE.cl::sycl::vec" @_Z29__spirv_SpecConstantCompositeDv2_i(<2 x i32> %[[#BV]])
;
; CHECK: %[[#POD:]] = call %struct._ZTS3POD.POD @"_Z29__spirv_SpecConstantCompositeAstruct._ZTS1A.Aclass._ZTSN2cl4sycl3vecIiLi2EEE.cl::sycl::vec"([2 x %struct._ZTS1A.A] %[[#NA]], %"class._ZTSN2cl4sycl3vecIiLi2EEE.cl::sycl::vec" %[[#B]])
; CHECK: store %struct._ZTS3POD.POD %[[#POD]]

; CHECK-LABEL: define {{.*}} spir_kernel void @_ZTS17SpecializedKernel
; CHECK: %[[#N0:]] = call i32 @_Z20__spirv_SpecConstantii(i32 [[#ID + 6]], i32
; CHECK: %[[#N1:]] = call float @_Z20__spirv_SpecConstantif(i32 [[#ID + 7]], float
; CHECK: %[[#CONST:]] = call %struct._ZTS1A.A @_Z29__spirv_SpecConstantCompositeif(i32 %[[#N0]], float %[[#N1]])
; CHECK: %struct._ZTS1A.A %[[#CONST]]
;
; CHECK: !sycl.specialization-constants = !{![[#MD0:]], ![[#MD1:]]}
;
; CHECK: ![[#MD0]] = !{!"_ZTS3POD", i32 [[#ID]], i32 0, i32 4,
; CHECK-SAME: i32 [[#ID + 1]], i32 4, i32 4,
; CHECK-SAME: i32 [[#ID + 2]], i32 8, i32 4,
; CHECK-SAME: i32 [[#ID + 3]], i32 12, i32 4,
; CHECK-SAME: i32 [[#ID + 4]], i32 16, i32 4,
; CHECK-SAME: i32 [[#ID + 5]], i32 20, i32 4}

; CHECK: ![[#MD1]] = !{!"_ZTS13MyComposConst", i32 [[#ID + 6]], i32 0, i32 4,
; CHECK-SAME: i32 [[#ID + 7]], i32 4, i32 4}

target datalayout = "e-i64:64-v16:16-v24:32-v32:32-v48:64-v96:128-v192:256-v256:256-v512:512-v1024:1024-n8:16:32:64"
target triple = "spir64-unknown-unknown-sycldevice"

%struct._ZTS3POD.POD = type { [2 x %struct._ZTS1A.A], %"class._ZTSN2cl4sycl3vecIiLi2EEE.cl::sycl::vec" }
%struct._ZTS1A.A = type { i32, float }
%"class._ZTSN2cl4sycl3vecIiLi2EEE.cl::sycl::vec" = type { <2 x i32> }
%"class._ZTSN2cl4sycl5rangeILi1EEE.cl::sycl::range" = type { %"class._ZTSN2cl4sycl6detail5arrayILi1EEE.cl::sycl::detail::array" }
%"class._ZTSN2cl4sycl6detail5arrayILi1EEE.cl::sycl::detail::array" = type { [1 x i64] }
%"class._ZTSN2cl4sycl2idILi1EEE.cl::sycl::id" = type { %"class._ZTSN2cl4sycl6detail5arrayILi1EEE.cl::sycl::detail::array" }

$_ZTS4Test = comdat any
$_ZTS17SpecializedKernel = comdat any

@__builtin_unique_stable_name._ZNK2cl4sycl6ONEAPI12experimental13spec_constantI3PODS4_E3getIS4_EENSt9enable_ifIXsr3std6is_podIT_EE5valueES8_E4typeEv = private unnamed_addr addrspace(1) constant [9 x i8] c"_ZTS3POD\00", align 1
@__builtin_unique_stable_name._ZNK2cl4sycl6ONEAPI12experimental13spec_constantI13MyComposConstE3getIS4_EENSt9enable_ifIXaasr3std8is_classIT_EE5valuesr3std6is_podIS9_EE5valueES9_E4typeEv = private unnamed_addr addrspace(1) constant [20 x i8] c"_ZTS13MyComposConst\00", align 1

; Function Attrs: convergent norecurse uwtable
define weak_odr dso_local spir_kernel void @_ZTS4Test(%struct._ZTS3POD.POD addrspace(1)* %_arg_, %"class._ZTSN2cl4sycl5rangeILi1EEE.cl::sycl::range"* byval(%"class._ZTSN2cl4sycl5rangeILi1EEE.cl::sycl::range") align 8 %_arg_1, %"class._ZTSN2cl4sycl5rangeILi1EEE.cl::sycl::range"* byval(%"class._ZTSN2cl4sycl5rangeILi1EEE.cl::sycl::range") align 8 %_arg_2, %"class._ZTSN2cl4sycl2idILi1EEE.cl::sycl::id"* byval(%"class._ZTSN2cl4sycl2idILi1EEE.cl::sycl::id") align 8 %_arg_3) local_unnamed_addr #0 comdat !kernel_arg_buffer_location !4 {
entry:
  %ref.tmp.i = alloca %struct._ZTS3POD.POD, align 8
  %0 = getelementptr inbounds %"class._ZTSN2cl4sycl2idILi1EEE.cl::sycl::id", %"class._ZTSN2cl4sycl2idILi1EEE.cl::sycl::id"* %_arg_3, i64 0, i32 0, i32 0, i64 0
  %1 = load i64, i64* %0, align 8
  %add.ptr.i = getelementptr inbounds %struct._ZTS3POD.POD, %struct._ZTS3POD.POD addrspace(1)* %_arg_, i64 %1
  %2 = bitcast %struct._ZTS3POD.POD* %ref.tmp.i to i8*
  call void @llvm.lifetime.start.p0i8(i64 24, i8* nonnull %2) #3
  %3 = addrspacecast %struct._ZTS3POD.POD* %ref.tmp.i to %struct._ZTS3POD.POD addrspace(4)*
  call spir_func void @_Z36__sycl_getCompositeSpecConstantValueI3PODET_PKc(%struct._ZTS3POD.POD addrspace(4)* sret(%struct._ZTS3POD.POD) align 8 %3, i8 addrspace(4)* addrspacecast (i8 addrspace(1)* getelementptr inbounds ([9 x i8], [9 x i8] addrspace(1)* @__builtin_unique_stable_name._ZNK2cl4sycl6ONEAPI12experimental13spec_constantI3PODS4_E3getIS4_EENSt9enable_ifIXsr3std6is_podIT_EE5valueES8_E4typeEv, i64 0, i64 0) to i8 addrspace(4)*)) #4
  %4 = bitcast %struct._ZTS3POD.POD addrspace(1)* %add.ptr.i to i8 addrspace(1)*
  %5 = addrspacecast i8 addrspace(1)* %4 to i8 addrspace(4)*
  call void @llvm.memcpy.p4i8.p0i8.i64(i8 addrspace(4)* align 8 dereferenceable(24) %5, i8* nonnull align 8 dereferenceable(24) %2, i64 24, i1 false), !tbaa.struct !5
  call void @llvm.lifetime.end.p0i8(i64 24, i8* nonnull %2) #3
  ret void
}

; Function Attrs: convergent norecurse
define weak_odr dso_local spir_kernel void @_ZTS17SpecializedKernel(float addrspace(1)* %_arg_, %"class._ZTSN2cl4sycl5rangeILi1EEE.cl::sycl::range"* byval(%"class._ZTSN2cl4sycl5rangeILi1EEE.cl::sycl::range") align 8 %_arg_1, %"class._ZTSN2cl4sycl5rangeILi1EEE.cl::sycl::range"* byval(%"class._ZTSN2cl4sycl5rangeILi1EEE.cl::sycl::range") align 8 %_arg_2, %"class._ZTSN2cl4sycl2idILi1EEE.cl::sycl::id"* byval(%"class._ZTSN2cl4sycl2idILi1EEE.cl::sycl::id") align 8 %_arg_3) local_unnamed_addr #0 comdat !kernel_arg_buffer_location !4 {
entry:
  %c.i = alloca %struct._ZTS1A.A, align 4
  %0 = getelementptr inbounds %"class._ZTSN2cl4sycl2idILi1EEE.cl::sycl::id", %"class._ZTSN2cl4sycl2idILi1EEE.cl::sycl::id"* %_arg_3, i64 0, i32 0, i32 0, i64 0
  %1 = addrspacecast i64* %0 to i64 addrspace(4)*
  %2 = load i64, i64 addrspace(4)* %1, align 8
  %add.ptr.i = getelementptr inbounds float, float addrspace(1)* %_arg_, i64 %2
  %c.ascast.i = addrspacecast %struct._ZTS1A.A* %c.i to %struct._ZTS1A.A addrspace(4)*
  %3 = bitcast %struct._ZTS1A.A* %c.i to i8*
  call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %3) #3
  call spir_func void @_Z40__sycl_getComposite2020SpecConstantValueI13MyComposConstET_PKcPvS4_(%struct._ZTS1A.A addrspace(4)* sret(%struct._ZTS1A.A) align 4 %c.ascast.i, i8 addrspace(4)* getelementptr inbounds ([20 x i8], [20 x i8] addrspace(4)* addrspacecast ([20 x i8] addrspace(1)* @__builtin_unique_stable_name._ZNK2cl4sycl6ONEAPI12experimental13spec_constantI13MyComposConstE3getIS4_EENSt9enable_ifIXaasr3std8is_classIT_EE5valuesr3std6is_podIS9_EE5valueES9_E4typeEv to [20 x i8] addrspace(4)*), i64 0, i64 0), i8 addrspace(4)* null, i8 addrspace(4)* null) #4
  %a.i = getelementptr inbounds %struct._ZTS1A.A, %struct._ZTS1A.A addrspace(4)* %c.ascast.i, i64 0, i32 0
  %4 = load i32, i32 addrspace(4)* %a.i, align 4
  %conv.i = sitofp i32 %4 to float
  %b.i = getelementptr inbounds %struct._ZTS1A.A, %struct._ZTS1A.A addrspace(4)* %c.ascast.i, i64 0, i32 1
  %5 = load float, float addrspace(4)* %b.i, align 4
  %add.i = fadd float %5, %conv.i
  %ptridx.ascast.i.i = addrspacecast float addrspace(1)* %add.ptr.i to float addrspace(4)*
  store float %add.i, float addrspace(4)* %ptridx.ascast.i.i, align 4, !tbaa !11
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %3) #3
  ret void
}

; Function Attrs: argmemonly nounwind willreturn
declare void @llvm.lifetime.start.p0i8(i64 immarg, i8* nocapture) #1

; Function Attrs: argmemonly nounwind willreturn
declare void @llvm.lifetime.end.p0i8(i64 immarg, i8* nocapture) #1

; Function Attrs: argmemonly nounwind willreturn
declare void @llvm.memcpy.p4i8.p0i8.i64(i8 addrspace(4)* noalias nocapture writeonly, i8* noalias nocapture readonly, i64, i1 immarg) #1

; Function Attrs: convergent
declare dso_local spir_func void @_Z36__sycl_getCompositeSpecConstantValueI3PODET_PKc(%struct._ZTS3POD.POD addrspace(4)* sret(%struct._ZTS3POD.POD) align 8, i8 addrspace(4)*) local_unnamed_addr #2

; Function Attrs: convergent
declare dso_local spir_func void @_Z40__sycl_getComposite2020SpecConstantValueI13MyComposConstET_PKcPvS4_(%struct._ZTS1A.A addrspace(4)* sret(%struct._ZTS1A.A) align 4, i8 addrspace(4)*, i8 addrspace(4)*, i8 addrspace(4)*) local_unnamed_addr #2

attributes #0 = { convergent norecurse uwtable "disable-tail-calls"="false" "frame-pointer"="all" "less-precise-fpmad"="false" "min-legal-vector-width"="0" "no-infs-fp-math"="false" "no-jump-tables"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "sycl-module-id"="../sycl/test/spec_const/composite.cpp" "tune-cpu"="generic" "uniform-work-group-size"="true" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #1 = { argmemonly nounwind willreturn }
attributes #2 = { convergent "disable-tail-calls"="false" "frame-pointer"="all" "less-precise-fpmad"="false" "no-infs-fp-math"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "tune-cpu"="generic" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #3 = { nounwind }
attributes #4 = { convergent }

!llvm.module.flags = !{!0}
!opencl.spir.version = !{!1}
!spirv.Source = !{!2}
!llvm.ident = !{!3}

!0 = !{i32 1, !"wchar_size", i32 4}
!1 = !{i32 1, i32 2}
!2 = !{i32 4, i32 100000}
!3 = !{!"clang version 12.0.0 "}
!4 = !{i32 -1, i32 -1, i32 -1, i32 -1}
!5 = !{i64 0, i64 16, !6, i64 16, i64 8, !6}
!6 = !{!7, !7, i64 0}
!7 = !{!"omnipotent char", !8, i64 0}
!8 = !{!"Simple C++ TBAA"}
!9 = !{!"int", !7, i64 0}
!10 = !{!"float", !7, i64 0}
!11 = !{!10, !10, i64 0}
