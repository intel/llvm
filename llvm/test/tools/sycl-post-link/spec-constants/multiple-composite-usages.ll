; RUN: sycl-post-link -spec-const=rt --ir-output-only %s -S -o - \
; RUN: | FileCheck %s --implicit-check-not "call {{.*}} __sycl_getCompositeSpecConstantValue"
;
; This test is intended to check that sycl-post-link tool is capable of handling
; situations when the same composite specialization constants is used more than
; once
;
; CHECK-LABEL: @foo1
; CHECK: call %struct._ZTS3POD.POD @"_Z29__spirv_SpecConstantCompositeAstruct._ZTS1A.Aclass._ZTSN2cl4sycl3vecIiLi2EEE.cl::sycl::vec"({{.*}})
; CHECK-LABEL: @_ZTS4Test
; CHECK: call %struct._ZTS3POD.POD @"_Z29__spirv_SpecConstantCompositeAstruct._ZTS1A.Aclass._ZTSN2cl4sycl3vecIiLi2EEE.cl::sycl::vec"({{.*}})
; CHECK-LABEL: @foo2
; CHECK: call %struct._ZTS3POD.POD @"_Z29__spirv_SpecConstantCompositeAstruct._ZTS1A.Aclass._ZTSN2cl4sycl3vecIiLi2EEE.cl::sycl::vec"({{.*}})

; CHECK: !sycl.specialization-constants = !{![[#MD0:]], ![[#MD1:]]
;
; CHECK-DAG: ![[#MD0]] = !{!"_ZTS3PO2", i32 [[#ID:]], i32 0, i32 4,
; CHECK-SAME: i32 [[#ID + 1]], i32 4, i32 4,
; CHECK-SAME: i32 [[#ID + 2]], i32 8, i32 4,
; CHECK-SAME: i32 [[#ID + 3]], i32 12, i32 4,
; CHECK-SAME: i32 [[#ID + 4]], i32 16, i32 4,
; CHECK-SAME: i32 [[#ID + 5]], i32 20, i32 4}
; CHECK-DAG: ![[#MD1]] = !{!"_ZTS3POD", i32 [[#ID1:]], i32 0, i32 4,
; CHECK-SAME: i32 [[#ID1 + 1]], i32 4, i32 4,
; CHECK-SAME: i32 [[#ID1 + 2]], i32 8, i32 4,
; CHECK-SAME: i32 [[#ID1 + 3]], i32 12, i32 4,
; CHECK-SAME: i32 [[#ID1 + 4]], i32 16, i32 4,
; CHECK-SAME: i32 [[#ID1 + 5]], i32 20, i32 4}

target datalayout = "e-i64:64-v16:16-v24:32-v32:32-v48:64-v96:128-v192:256-v256:256-v512:512-v1024:1024-n8:16:32:64"
target triple = "spir64-unknown-unknown-sycldevice"

%struct._ZTS3POD.POD = type { [2 x %struct._ZTS1A.A], %"class._ZTSN2cl4sycl3vecIiLi2EEE.cl::sycl::vec" }
%struct._ZTS1A.A = type { i32, float }
%"class._ZTSN2cl4sycl3vecIiLi2EEE.cl::sycl::vec" = type { <2 x i32> }
%"class._ZTSN2cl4sycl5rangeILi1EEE.cl::sycl::range" = type { %"class._ZTSN2cl4sycl6detail5arrayILi1EEE.cl::sycl::detail::array" }
%"class._ZTSN2cl4sycl6detail5arrayILi1EEE.cl::sycl::detail::array" = type { [1 x i64] }
%"class._ZTSN2cl4sycl2idILi1EEE.cl::sycl::id" = type { %"class._ZTSN2cl4sycl6detail5arrayILi1EEE.cl::sycl::detail::array" }

$_ZTS4Test = comdat any

@__builtin_unique_stable_name._ZNK2cl4sycl6ONEAPI12experimental13spec_constantI3PODS4_E3getIS4_EENSt9enable_ifIXsr3std6is_podIT_EE5valueES8_E4typeEv = private unnamed_addr addrspace(1) constant [9 x i8] c"_ZTS3POD\00", align 1
@__builtin_unique_stable_name.2 = private unnamed_addr addrspace(1) constant [9 x i8] c"_ZTS3PO2\00", align 1

define spir_func void @foo1() {
  %ref.tmp.i = alloca %struct._ZTS3POD.POD, align 8
  %1 = addrspacecast %struct._ZTS3POD.POD* %ref.tmp.i to %struct._ZTS3POD.POD addrspace(4)*
  call spir_func void @_Z36__sycl_getCompositeSpecConstantValueI3PODET_PKc(%struct._ZTS3POD.POD addrspace(4)* sret(%struct._ZTS3POD.POD) align 8 %1, i8 addrspace(4)* addrspacecast (i8 addrspace(1)* getelementptr inbounds ([9 x i8], [9 x i8] addrspace(1)* @__builtin_unique_stable_name.2, i64 0, i64 0) to i8 addrspace(4)*)) #4
  ret void
}

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

define spir_func void @foo2() {
  %ref.tmp.i = alloca %struct._ZTS3POD.POD, align 8
  %1 = addrspacecast %struct._ZTS3POD.POD* %ref.tmp.i to %struct._ZTS3POD.POD addrspace(4)*
  call spir_func void @_Z36__sycl_getCompositeSpecConstantValueI3PODET_PKc(%struct._ZTS3POD.POD addrspace(4)* sret(%struct._ZTS3POD.POD) align 8 %1, i8 addrspace(4)* addrspacecast (i8 addrspace(1)* getelementptr inbounds ([9 x i8], [9 x i8] addrspace(1)* @__builtin_unique_stable_name.2, i64 0, i64 0) to i8 addrspace(4)*)) #4
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
