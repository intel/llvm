; RUN: sycl-post-link --ir-output-only --spec-const=native %s -S -o - | FileCheck %s
; RUN: sycl-post-link -debug-only=SpecConst --spec-const=native %s -S 2>&1 | FileCheck %s --check-prefix=CHECK-LOG
;
; This test is intended to check that SpecConstantsPass is able to handle the
; situation where specialization constants have zeroinitializer in LLVM IR

target datalayout = "e-i64:64-v16:16-v24:32-v32:32-v48:64-v96:128-v192:256-v256:256-v512:512-v1024:1024-n8:16:32:64"
target triple = "spir64-unknown-unknown"

%"class._ZTSN2cl4sycl17specialization_idIiEE.cl::sycl::specialization_id" = type { i32 }
%"class._ZTSN2cl4sycl17specialization_idIdEE.cl::sycl::specialization_id" = type { double }
%"class._ZTSN2cl4sycl17specialization_idI9compositeEE.cl::sycl::specialization_id" = type { %struct._ZTS9composite.composite }
%struct._ZTS9composite.composite = type { %struct._ZTS6nested.nested, i8, i8, i64 }
%struct._ZTS6nested.nested = type { float }
%"class._ZTSN2cl4sycl5rangeILi1EEE.cl::sycl::range" = type { %"class._ZTSN2cl4sycl6detail5arrayILi1EEE.cl::sycl::detail::array" }
%"class._ZTSN2cl4sycl6detail5arrayILi1EEE.cl::sycl::detail::array" = type { [1 x i64] }
%"class._ZTSN2cl4sycl2idILi1EEE.cl::sycl::id" = type { %"class._ZTSN2cl4sycl6detail5arrayILi1EEE.cl::sycl::detail::array" }

$"_ZTSZZ4mainENK3$_0clERN2cl4sycl7handlerEE7Kernel1" = comdat any

@__builtin_unique_stable_name._ZN2cl4sycl14kernel_handler33getSpecializationConstantOnDeviceIL_ZL6int_idEiLPv0EEET0_v = private unnamed_addr addrspace(1) constant [70 x i8] c"_ZTSN2cl4sycl6detail32specialization_id_name_generatorIL_ZL6int_idEEE\00", align 1
@_ZL6int_id = internal addrspace(1) constant %"class._ZTSN2cl4sycl17specialization_idIiEE.cl::sycl::specialization_id" zeroinitializer, align 4
@__builtin_unique_stable_name._ZN2cl4sycl14kernel_handler33getSpecializationConstantOnDeviceIL_ZL9double_idEdLPv0EEET0_v = private unnamed_addr addrspace(1) constant [73 x i8] c"_ZTSN2cl4sycl6detail32specialization_id_name_generatorIL_ZL9double_idEEE\00", align 1
@_ZL9double_id = internal addrspace(1) constant %"class._ZTSN2cl4sycl17specialization_idIdEE.cl::sycl::specialization_id" zeroinitializer, align 8
@__builtin_unique_stable_name._ZN2cl4sycl14kernel_handler33getSpecializationConstantOnDeviceIL_ZL12composite_idE9compositeLPv0EEET0_v = private unnamed_addr addrspace(1) constant [77 x i8] c"_ZTSN2cl4sycl6detail32specialization_id_name_generatorIL_ZL12composite_idEEE\00", align 1
@_ZL12composite_id = internal addrspace(1) constant %"class._ZTSN2cl4sycl17specialization_idI9compositeEE.cl::sycl::specialization_id" zeroinitializer, align 8

; Function Attrs: convergent norecurse
define weak_odr dso_local spir_kernel void @"_ZTSZZ4mainENK3$_0clERN2cl4sycl7handlerEE7Kernel1"() local_unnamed_addr #0 comdat {
entry:
  %ref.tmp.i = alloca %struct._ZTS9composite.composite, align 8
  %ref.tmp.ascast.i = addrspacecast %struct._ZTS9composite.composite* %ref.tmp.i to %struct._ZTS9composite.composite addrspace(4)*
  %call.i.i.i = tail call spir_func i32 @_Z37__sycl_getScalar2020SpecConstantValueIiET_PKcPKvS4_(i8 addrspace(4)* getelementptr inbounds ([70 x i8], [70 x i8] addrspace(4)* addrspacecast ([70 x i8] addrspace(1)* @__builtin_unique_stable_name._ZN2cl4sycl14kernel_handler33getSpecializationConstantOnDeviceIL_ZL6int_idEiLPv0EEET0_v to [70 x i8] addrspace(4)*), i64 0, i64 0), i8 addrspace(4)* addrspacecast (i8 addrspace(1)* bitcast (%"class._ZTSN2cl4sycl17specialization_idIiEE.cl::sycl::specialization_id" addrspace(1)* @_ZL6int_id to i8 addrspace(1)*) to i8 addrspace(4)*), i8 addrspace(4)* null) #3
; CHECK: call i32 @_Z20__spirv_SpecConstantii(i32 [[#SCID0:]], i32 0)

  %call.i.i23.i = tail call spir_func double @_Z37__sycl_getScalar2020SpecConstantValueIdET_PKcPKvS4_(i8 addrspace(4)* getelementptr inbounds ([73 x i8], [73 x i8] addrspace(4)* addrspacecast ([73 x i8] addrspace(1)* @__builtin_unique_stable_name._ZN2cl4sycl14kernel_handler33getSpecializationConstantOnDeviceIL_ZL9double_idEdLPv0EEET0_v to [73 x i8] addrspace(4)*), i64 0, i64 0), i8 addrspace(4)* addrspacecast (i8 addrspace(1)* bitcast (%"class._ZTSN2cl4sycl17specialization_idIdEE.cl::sycl::specialization_id" addrspace(1)* @_ZL9double_id to i8 addrspace(1)*) to i8 addrspace(4)*), i8 addrspace(4)* null) #3
; CHECK: call double @_Z20__spirv_SpecConstantid(i32 [[#SCID1:]], double 0.000000e+00)

  call spir_func void @_Z40__sycl_getComposite2020SpecConstantValueI9compositeET_PKcPKvS5_(%struct._ZTS9composite.composite addrspace(4)* sret(%struct._ZTS9composite.composite) align 8 %ref.tmp.ascast.i, i8 addrspace(4)* getelementptr inbounds ([77 x i8], [77 x i8] addrspace(4)* addrspacecast ([77 x i8] addrspace(1)* @__builtin_unique_stable_name._ZN2cl4sycl14kernel_handler33getSpecializationConstantOnDeviceIL_ZL12composite_idE9compositeLPv0EEET0_v to [77 x i8] addrspace(4)*), i64 0, i64 0), i8 addrspace(4)* addrspacecast (i8 addrspace(1)* bitcast (%"class._ZTSN2cl4sycl17specialization_idI9compositeEE.cl::sycl::specialization_id" addrspace(1)* @_ZL12composite_id to i8 addrspace(1)*) to i8 addrspace(4)*), i8 addrspace(4)* null) #3
; CHECK: call float @_Z20__spirv_SpecConstantif(i32 [[#SCID2:]], float 0.000000e+00)
; CHECK: call i8 @_Z20__spirv_SpecConstantia(i32 [[#SCID3:]], i8 0)
; CHECK: call i8 @_Z20__spirv_SpecConstantia(i32 [[#SCID4:]], i8 0)
; CHECK: call i64 @_Z20__spirv_SpecConstantix(i32 [[#SCID5:]], i64 0)

; CHECK-LOG: sycl.specialization-constants
; CHECK-LOG:[[UNIQUE_PREFIX:[0-9a-zA-Z]+]]={0, 0, 4}
; CHECK-LOG:[[UNIQUE_PREFIX2:[0-9a-zA-Z]+]]={1, 0, 8}
; CHECK-LOG:[[UNIQUE_PREFIX3:[0-9a-zA-Z]+]]={2, 0, 4}
; CHECK-LOG:[[UNIQUE_PREFIX3]]={3, 4, 1}
; CHECK-LOG:[[UNIQUE_PREFIX3]]={4, 5, 1}
; CHECK-LOG:[[UNIQUE_PREFIX3]]={5, 8, 8}
; CHECK-LOG: sycl.specialization-constants-default-values
; CHECK-LOG: {0, 4, 0}
; CHECK-LOG: {4, 8, 0.000000e+00}
; CHECK-LOG: {12, 16, 0}

  ret void
}

; Function Attrs: convergent
declare dso_local spir_func i32 @_Z37__sycl_getScalar2020SpecConstantValueIiET_PKcPKvS4_(i8 addrspace(4)*, i8 addrspace(4)*, i8 addrspace(4)*) local_unnamed_addr #2

; Function Attrs: convergent
declare dso_local spir_func double @_Z37__sycl_getScalar2020SpecConstantValueIdET_PKcPKvS4_(i8 addrspace(4)*, i8 addrspace(4)*, i8 addrspace(4)*) local_unnamed_addr #2

; Function Attrs: convergent
declare dso_local spir_func void @_Z40__sycl_getComposite2020SpecConstantValueI9compositeET_PKcPKvS5_(%struct._ZTS9composite.composite addrspace(4)* sret(%struct._ZTS9composite.composite) align 8, i8 addrspace(4)*, i8 addrspace(4)*, i8 addrspace(4)*) local_unnamed_addr #2

attributes #0 = { convergent norecurse "frame-pointer"="all" "min-legal-vector-width"="0" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "sycl-module-id"="t.cpp" "uniform-work-group-size"="true" }
attributes #1 = { argmemonly nofree nosync nounwind willreturn }
attributes #2 = { convergent "frame-pointer"="all" "no-trapping-math"="true" "stack-protector-buffer-size"="8" }
attributes #3 = { convergent }
attributes #4 = { nounwind }

!llvm.module.flags = !{!0}
!opencl.spir.version = !{!1}
!spirv.Source = !{!2}
!llvm.ident = !{!3}

!0 = !{i32 1, !"wchar_size", i32 4}
!1 = !{i32 1, i32 2}
!2 = !{i32 4, i32 100000}
!3 = !{!"clang version 13.0.0 (/data/github.com/intel/llvm/clang c1f0fd875de45645c66776667682af55059968b7)"}
!4 = !{i32 -1, i32 -1, i32 -1, i32 -1}
!5 = !{!6, !6, i64 0}
!6 = !{!"int", !7, i64 0}
!7 = !{!"omnipotent char", !8, i64 0}
!8 = !{!"Simple C++ TBAA"}
!9 = !{!10, !7, i64 5}
!10 = !{!"_ZTS9composite", !11, i64 0, !13, i64 4, !7, i64 5, !14, i64 8}
!11 = !{!"_ZTS6nested", !12, i64 0}
!12 = !{!"float", !7, i64 0}
!13 = !{!"bool", !7, i64 0}
!14 = !{!"long long", !7, i64 0}
