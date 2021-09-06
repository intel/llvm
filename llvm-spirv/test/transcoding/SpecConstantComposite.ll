; RUN: llvm-as < %s -o %t.bc
; RUN: llvm-spirv %t.bc -spirv-text -o %t.spt
; RUN: FileCheck < %t.spt %s --check-prefix=CHECK-SPIRV
; RUN: llvm-spirv %t.bc -o %t.spv
; RUN: spirv-val %t.spv
; RUN: llvm-spirv %t.spv -r -o - | llvm-dis -o %t.ll
; RUN: FileCheck < %t.ll %s --check-prefix=CHECK-LLVM

; CHECK-SPIRV: Decorate [[#SC3:]] SpecId 3
; CHECK-SPIRV: Decorate [[#SC4:]] SpecId 4
; CHECK-SPIRV: Decorate [[#SC6:]] SpecId 6
; CHECK-SPIRV: Decorate [[#SC7:]] SpecId 7
; CHECK-SPIRV: Decorate [[#SC10:]] SpecId 10
; CHECK-SPIRV: Decorate [[#SC11:]] SpecId 11

; CHECK-SPIRV-DAG: TypeInt [[#Int:]] 32
; CHECK-SPIRV-DAG: TypeFloat [[#Float:]] 32
; CHECK-SPIRV-DAG: TypeStruct [[#StructA:]] [[#Int]] [[#Float]]
; CHECK-SPIRV-DAG: TypeArray [[#Array:]] [[#StructA]] [[#]]
; CHECK-SPIRV-DAG: TypeVector [[#Vector:]] [[#Int]] 2
; CHECK-SPIRV-DAG: TypeStruct [[#Struct:]] [[#Vector]]
; CHECK-SPIRV-DAG: TypeStruct [[#POD_TYPE:]] [[#Array]] [[#Struct]]

source_filename = "./SpecConstantComposite.ll"
target datalayout = "e-i64:64-v16:16-v24:32-v32:32-v48:64-v96:128-v192:256-v256:256-v512:512-v1024:1024-n8:16:32:64"
target triple = "spir64-unknown-unknown"

%struct._ZTS3POD.POD = type { [2 x %struct._ZTS1A.A], %"class._ZTSN2cl4sycl3vecIiLi2EEE.cl::sycl::vec" }
%struct._ZTS1A.A = type { i32, float }
%"class._ZTSN2cl4sycl3vecIiLi2EEE.cl::sycl::vec" = type { <2 x i32> }
%"class._ZTSN2cl4sycl5rangeILi1EEE.cl::sycl::range" = type { %"class._ZTSN2cl4sycl6detail5arrayILi1EEE.cl::sycl::detail::array" }
%"class._ZTSN2cl4sycl6detail5arrayILi1EEE.cl::sycl::detail::array" = type { [1 x i64] }
%"class._ZTSN2cl4sycl2idILi1EEE.cl::sycl::id" = type { %"class._ZTSN2cl4sycl6detail5arrayILi1EEE.cl::sycl::detail::array" }

$_ZTS4Test = comdat any

; Function Attrs: convergent norecurse uwtable
define weak_odr dso_local spir_kernel void @_ZTS4Test(%struct._ZTS3POD.POD addrspace(1)* %_arg_, %"class._ZTSN2cl4sycl5rangeILi1EEE.cl::sycl::range"* byval(%"class._ZTSN2cl4sycl5rangeILi1EEE.cl::sycl::range") align 8 %_arg_1, %"class._ZTSN2cl4sycl5rangeILi1EEE.cl::sycl::range"* byval(%"class._ZTSN2cl4sycl5rangeILi1EEE.cl::sycl::range") align 8 %_arg_2, %"class._ZTSN2cl4sycl2idILi1EEE.cl::sycl::id"* byval(%"class._ZTSN2cl4sycl2idILi1EEE.cl::sycl::id") align 8 %_arg_3) local_unnamed_addr #0 comdat !kernel_arg_buffer_location !4 {
entry:
  %ref.tmp.i = alloca %struct._ZTS3POD.POD, align 8
  %0 = getelementptr inbounds %"class._ZTSN2cl4sycl2idILi1EEE.cl::sycl::id", %"class._ZTSN2cl4sycl2idILi1EEE.cl::sycl::id"* %_arg_3, i64 0, i32 0, i32 0, i64 0
  %1 = load i64, i64* %0, align 8
  %add.ptr.i = getelementptr inbounds %struct._ZTS3POD.POD, %struct._ZTS3POD.POD addrspace(1)* %_arg_, i64 %1
  %2 = bitcast %struct._ZTS3POD.POD* %ref.tmp.i to i8*
  call void @llvm.lifetime.start.p0i8(i64 24, i8* nonnull %2) #2
  %3 = addrspacecast %struct._ZTS3POD.POD* %ref.tmp.i to %struct._ZTS3POD.POD addrspace(4)*

  %4 = call i32 @_Z20__spirv_SpecConstantii(i32 3, i32 1)
; CHECK-SPIRV-DAG: SpecConstant [[#Int]] [[#SC3]] 1

  %5 = call float @_Z20__spirv_SpecConstantif(i32 4, float 0.000000e+00)
; CHECK-SPIRV-DAG: SpecConstant [[#Float]] [[#SC4]] 0

  %6 = call %struct._ZTS1A.A @_Z29__spirv_SpecConstantCompositeif(i32 %4, float %5)
; CHECK-SPIRV-DAG: SpecConstantComposite [[#StructA]] [[#SC_StructA0:]] [[#SC3]] [[#SC4]]

  %7 = call i32 @_Z20__spirv_SpecConstantii(i32 6, i32 35)
; CHECK-SPIRV-DAG: SpecConstant [[#Int]] [[#SC6]] 35

  %8 = call float @_Z20__spirv_SpecConstantif(i32 7, float 0.000000e+00)
; CHECK-SPIRV-DAG: SpecConstant [[#Float]] [[#SC7]] 0

  %9 = call %struct._ZTS1A.A @_Z29__spirv_SpecConstantCompositeif(i32 %7, float %8)
; CHECK-SPIRV-DAG: SpecConstantComposite [[#StructA]] [[#SC_StructA1:]] [[#SC6]] [[#SC7]]

  %10 = call [2 x %struct._ZTS1A.A] @_Z29__spirv_SpecConstantCompositestruct._ZTS1A.Astruct._ZTS1A.A(%struct._ZTS1A.A %6, %struct._ZTS1A.A %9)
; CHECK-SPIRV-DAG: SpecConstantComposite [[#Array]] [[#SC_Array:]] [[#SC_StructA0]] [[#SC_StructA1]]

  %11 = call i32 @_Z20__spirv_SpecConstantii(i32 10, i32 45)
; CHECK-SPIRV-DAG: SpecConstant [[#Int]] [[#SC10]] 45

  %12 = call i32 @_Z20__spirv_SpecConstantii(i32 11, i32 55)
; CHECK-SPIRV-DAG: SpecConstant [[#Int]] [[#SC11]] 55

  %13 = call <2 x i32> @_Z29__spirv_SpecConstantCompositeii(i32 %11, i32 %12)
; CHECK-SPIRV-DAG: SpecConstantComposite [[#Vector]] [[#SC_Vector:]] [[#SC10]] [[#SC11]]

  %14 = call %"class._ZTSN2cl4sycl3vecIiLi2EEE.cl::sycl::vec" @_Z29__spirv_SpecConstantCompositeDv2_i(<2 x i32> %13)
; CHECK-SPIRV-DAG: SpecConstantComposite [[#Struct]] [[#SC_Struct:]] [[#SC_Vector]]

  %15 = call %struct._ZTS3POD.POD @"_Z29__spirv_SpecConstantCompositeAstruct._ZTS1A.Aclass._ZTSN2cl4sycl3vecIiLi2EEE.cl::sycl::vec"([2 x %struct._ZTS1A.A] %10, %"class._ZTSN2cl4sycl3vecIiLi2EEE.cl::sycl::vec" %14), !SYCL_SPEC_CONST_SYM_ID !5
; CHECK-SPIRV-DAG: SpecConstantComposite [[#POD_TYPE]] [[#SC_POD:]] [[#SC_Array]] [[#SC_Struct]]

  store %struct._ZTS3POD.POD %15, %struct._ZTS3POD.POD addrspace(4)* %3, align 8
; CHECK-SPIRV-DAG: Store [[#]] [[#SC_POD]]
; CHECK-LLVM: store %struct._ZTS3POD.POD { [2 x %struct._ZTS1A.A] [%struct._ZTS1A.A { i32 1, float 0.000000e+00 }, %struct._ZTS1A.A { i32 35, float 0.000000e+00 }], %"class._ZTSN2cl4sycl3vecIiLi2EEE.cl::sycl::vec" { <2 x i32> <i32 45, i32 55> } }

  %16 = bitcast %struct._ZTS3POD.POD addrspace(1)* %add.ptr.i to i8 addrspace(1)*
  %17 = addrspacecast i8 addrspace(1)* %16 to i8 addrspace(4)*
  call void @llvm.memcpy.p4i8.p0i8.i64(i8 addrspace(4)* align 8 dereferenceable(24) %17, i8* nonnull align 8 dereferenceable(24) %2, i64 24, i1 false), !tbaa.struct !6
  call void @llvm.lifetime.end.p0i8(i64 24, i8* nonnull %2) #2
  ret void
}

; Function Attrs: argmemonly nounwind willreturn
declare void @llvm.lifetime.start.p0i8(i64 immarg, i8* nocapture) #1

; Function Attrs: argmemonly nounwind willreturn
declare void @llvm.lifetime.end.p0i8(i64 immarg, i8* nocapture) #1

; Function Attrs: argmemonly nounwind willreturn
declare void @llvm.memcpy.p4i8.p0i8.i64(i8 addrspace(4)* noalias nocapture writeonly, i8* noalias nocapture readonly, i64, i1 immarg) #1

declare i32 @_Z20__spirv_SpecConstantii(i32, i32)

declare float @_Z20__spirv_SpecConstantif(i32, float)

declare %struct._ZTS1A.A @_Z29__spirv_SpecConstantCompositeif(i32, float)

declare [2 x %struct._ZTS1A.A] @_Z29__spirv_SpecConstantCompositestruct._ZTS1A.Astruct._ZTS1A.A(%struct._ZTS1A.A, %struct._ZTS1A.A)

declare <2 x i32> @_Z29__spirv_SpecConstantCompositeii(i32, i32)

declare %"class._ZTSN2cl4sycl3vecIiLi2EEE.cl::sycl::vec" @_Z29__spirv_SpecConstantCompositeDv2_i(<2 x i32>)

declare %struct._ZTS3POD.POD @"_Z29__spirv_SpecConstantCompositeAstruct._ZTS1A.Aclass._ZTSN2cl4sycl3vecIiLi2EEE.cl::sycl::vec"([2 x %struct._ZTS1A.A], %"class._ZTSN2cl4sycl3vecIiLi2EEE.cl::sycl::vec")

attributes #0 = { convergent norecurse uwtable "disable-tail-calls"="false" "frame-pointer"="all" "less-precise-fpmad"="false" "min-legal-vector-width"="0" "no-infs-fp-math"="false" "no-jump-tables"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "sycl-module-id"="../sycl/test/spec_const/composite.cpp" "tune-cpu"="generic" "uniform-work-group-size"="true" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #1 = { argmemonly nounwind willreturn }
attributes #2 = { nounwind }

!llvm.module.flags = !{!0}
!opencl.spir.version = !{!1}
!spirv.Source = !{!2}
!llvm.ident = !{!3}

!0 = !{i32 1, !"wchar_size", i32 4}
!1 = !{i32 1, i32 2}
!2 = !{i32 4, i32 100000}
!3 = !{!"clang version 12.0.0 (/data/github.com/intel/llvm/clang 56ee5b054b5a1f2f703722fc414fcb05af18b40a)"}
!4 = !{i32 -1, i32 -1, i32 -1, i32 -1}
!5 = !{!"_ZTS3POD", i32 3, i32 4, i32 6, i32 7, i32 10, i32 11}
!6 = !{i64 0, i64 16, !7, i64 16, i64 8, !7}
!7 = !{!8, !8, i64 0}
!8 = !{!"omnipotent char", !9, i64 0}
!9 = !{!"Simple C++ TBAA"}
