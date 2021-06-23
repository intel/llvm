; RUN: sycl-post-link --ir-output-only -spec-const=default %s -S -o - | \
; RUN:   FileCheck %s -check-prefixes=CHECK,CHECK-DEF
; RUN: sycl-post-link --ir-output-only -spec-const=rt %s -S -o - | \
; RUN:   FileCheck %s -check-prefixes=CHECK,CHECK-RT

; This test checks that the post link tool is able to correctly transform
; specialization constant intrinsics for different types in a device code
; compiled with -O2 (This LLVM IR was optimized with 'opt -O2')

target datalayout = "e-i64:64-v16:16-v24:32-v32:32-v48:64-v96:128-v192:256-v256:256-v512:512-v1024:1024"
target triple = "spir64-unknown-linux-sycldevice"

%"cl::sycl::range" = type { %"cl::sycl::detail::array" }
%"cl::sycl::detail::array" = type { [1 x i64] }
%"cl::sycl::id" = type { %"cl::sycl::detail::array" }

$_ZTS17SpecializedKernel = comdat any

@__unique_stable_name.SC_Ib11MyBoolConstE3getEv = private unnamed_addr constant [18 x i8] c"_ZTS11MyBoolConst\00", align 1
@__unique_stable_name.SC_Ia11MyInt8ConstE3getEv = private unnamed_addr constant [18 x i8] c"_ZTS11MyInt8Const\00", align 1
@__unique_stable_name.SC_Ih12MyUInt8ConstE3getEv = private unnamed_addr constant [19 x i8] c"_ZTS12MyUInt8Const\00", align 1
@__unique_stable_name.SC_Is12MyInt16ConstE3getEv = private unnamed_addr constant [19 x i8] c"_ZTS12MyInt16Const\00", align 1
@__unique_stable_name.SC_It13MyUInt16ConstE3getEv = private unnamed_addr constant [20 x i8] c"_ZTS13MyUInt16Const\00", align 1
@__unique_stable_name.SC_Ii12MyInt32ConstE3getEv = private unnamed_addr constant [19 x i8] c"_ZTS12MyInt32Const\00", align 1
@__unique_stable_name.SC_Ij13MyUInt32ConstE3getEv = private unnamed_addr constant [20 x i8] c"_ZTS13MyUInt32Const\00", align 1
@__unique_stable_name.SC_Il12MyInt64ConstE3getEv = private unnamed_addr constant [19 x i8] c"_ZTS12MyInt64Const\00", align 1
@__unique_stable_name.SC_Im13MyUInt64ConstE3getEv = private unnamed_addr constant [20 x i8] c"_ZTS13MyUInt64Const\00", align 1
@__unique_stable_name.SC_If12MyFloatConstE3getEv = private unnamed_addr constant [19 x i8] c"_ZTS12MyFloatConst\00", align 1
@__unique_stable_name.SC_Id13MyDoubleConstE3getEv = private unnamed_addr constant [20 x i8] c"_ZTS13MyDoubleConst\00", align 1
@__unique_stable_name.SC_Id14MyDoubleConst2E3getEv = private unnamed_addr constant [21 x i8] c"_ZTS14MyDoubleConst2\00", align 1

; Function Attrs: norecurse
define weak_odr dso_local spir_kernel void @_ZTS17SpecializedKernel(float addrspace(1)* %0, %"cl::sycl::range"* byval(%"cl::sycl::range") align 8 %1, %"cl::sycl::range"* byval(%"cl::sycl::range") align 8 %2, %"cl::sycl::id"* byval(%"cl::sycl::id") align 8 %3) local_unnamed_addr #0 comdat !kernel_arg_addr_space !4 !kernel_arg_access_qual !5 !kernel_arg_type !6 !kernel_arg_base_type !6 !kernel_arg_type_qual !7 {
  %5 = getelementptr inbounds %"cl::sycl::id", %"cl::sycl::id"* %3, i64 0, i32 0, i32 0, i64 0
  %6 = load i64, i64* %5, align 8
  %7 = getelementptr inbounds float, float addrspace(1)* %0, i64 %6
  %8 = tail call spir_func zeroext i1 @_Z33__sycl_getScalarSpecConstantValueIbET_PKc(i8 addrspace(4)* addrspacecast (i8* getelementptr inbounds ([18 x i8], [18 x i8]* @__unique_stable_name.SC_Ib11MyBoolConstE3getEv, i64 0, i64 0) to i8 addrspace(4)*))
;;;;;;;;;;;;;; check that __sycl* intrinsic goes away:
; CHECK-NOT: %{{[0-9]+}} ={{.*}} call {{.*}}@_Z33__sycl_getScalarSpecConstantValueIbET_PKc
;;;;;;;;;;;;;; check that with -spec-const=rt __spirv* intrinsic is generated:
; CHECK-RT: %{{[0-9]+}} = call i1 @_Z20__spirv_SpecConstantib(i32 0, i1 false)
;;;;;;;;;;;;;; check that with -spec-const=default __spirv* intrinsic is not
;;;;;;;;;;;;;; generated:
; CHECK-DEF-NOT: %{{[0-9]+}} = call i1 @_Z20__spirv_SpecConstant
  %9 = zext i1 %8 to i32
;;;;;;;;;;;;;; check that with -spec-const=default values returns by __sycl*
;;;;;;;;;;;;;; intrinsics are replaced with constants:
; CHECK-DEF: %[[VAL0:[0-9]+]] = zext i1 false to i32
  %10 = tail call spir_func signext i8 @_Z33__sycl_getScalarSpecConstantValueIaET_PKc(i8 addrspace(4)* addrspacecast (i8* getelementptr inbounds ([18 x i8], [18 x i8]* @__unique_stable_name.SC_Ia11MyInt8ConstE3getEv, i64 0, i64 0) to i8 addrspace(4)*))
; CHECK-RT: %{{[0-9]+}} = call i8 @_Z20__spirv_SpecConstantia(i32 1, i8 0)
  %11 = sext i8 %10 to i32
; CHECK-DEF: %[[VAL1:[0-9]+]] = sext i8 0 to i32
  %12 = add nsw i32 %11, %9
; CHECK-DEF: %[[SUM0:[0-9]+]] = add nsw i32 %[[VAL1]], %[[VAL0]]
  %13 = tail call spir_func zeroext i8 @_Z33__sycl_getScalarSpecConstantValueIhET_PKc(i8 addrspace(4)* addrspacecast (i8* getelementptr inbounds ([19 x i8], [19 x i8]* @__unique_stable_name.SC_Ih12MyUInt8ConstE3getEv, i64 0, i64 0) to i8 addrspace(4)*))
; CHECK-RT: %{{[0-9]+}} = call i8 @_Z20__spirv_SpecConstantia(i32 2, i8 0)
  %14 = zext i8 %13 to i32
; CHECK-DEF: %[[VAL2:[0-9]+]] = zext i8 0 to i32
  %15 = add nsw i32 %12, %14
; CHECK-DEF: %[[SUM1:[0-9]+]] = add nsw i32 %[[SUM0]], %[[VAL2]]
  %16 = tail call spir_func signext i16 @_Z33__sycl_getScalarSpecConstantValueIsET_PKc(i8 addrspace(4)* addrspacecast (i8* getelementptr inbounds ([19 x i8], [19 x i8]* @__unique_stable_name.SC_Is12MyInt16ConstE3getEv, i64 0, i64 0) to i8 addrspace(4)*))
; CHECK-RT: %{{[0-9]+}} = call i16 @_Z20__spirv_SpecConstantis(i32 3, i16 0)
  %17 = sext i16 %16 to i32
; CHECK-DEF: %[[VAL3:[0-9]+]] = sext i16 0 to i32
  %18 = add nsw i32 %15, %17
; CHECK-DEF: %[[SUM2:[0-9]+]] = add nsw i32 %[[SUM1]], %[[VAL3]]
  %19 = tail call spir_func zeroext i16 @_Z33__sycl_getScalarSpecConstantValueItET_PKc(i8 addrspace(4)* addrspacecast (i8* getelementptr inbounds ([20 x i8], [20 x i8]* @__unique_stable_name.SC_It13MyUInt16ConstE3getEv, i64 0, i64 0) to i8 addrspace(4)*))
; CHECK-RT: %{{[0-9]+}} = call i16 @_Z20__spirv_SpecConstantis(i32 4, i16 0)
  %20 = zext i16 %19 to i32
; CHECK-DEF: %[[VAL4:[0-9]+]] = zext i16 0 to i32
  %21 = add nsw i32 %18, %20
; CHECK-DEF: %[[SUM3:[0-9]+]] = add nsw i32 %[[SUM2]], %[[VAL4]]
  %22 = tail call spir_func i32 @_Z33__sycl_getScalarSpecConstantValueIiET_PKc(i8 addrspace(4)* addrspacecast (i8* getelementptr inbounds ([19 x i8], [19 x i8]* @__unique_stable_name.SC_Ii12MyInt32ConstE3getEv, i64 0, i64 0) to i8 addrspace(4)*))
; CHECK-RT: %{{[0-9]+}} = call i32 @_Z20__spirv_SpecConstantii(i32 5, i32 0)
  %23 = add nsw i32 %21, %22
; CHECK-DEF: %[[SUM4:[0-9]+]] = add nsw i32 %[[SUM3]], 0
  %24 = tail call spir_func i32 @_Z33__sycl_getScalarSpecConstantValueIjET_PKc(i8 addrspace(4)* addrspacecast (i8* getelementptr inbounds ([20 x i8], [20 x i8]* @__unique_stable_name.SC_Ij13MyUInt32ConstE3getEv, i64 0, i64 0) to i8 addrspace(4)*))
; CHECK-RT: %{{[0-9]+}} = call i32 @_Z20__spirv_SpecConstantii(i32 6, i32 0)
  %25 = add i32 %23, %24
; CHECK-DEF: %[[SUM5:[0-9]+]] = add i32 %[[SUM4]], 0
  %26 = zext i32 %25 to i64
; CHECK-DEF: %[[VAL5:[0-9]+]] = zext i32 %[[SUM5]] to i64
  %27 = tail call spir_func i64 @_Z33__sycl_getScalarSpecConstantValueIlET_PKc(i8 addrspace(4)* addrspacecast (i8* getelementptr inbounds ([19 x i8], [19 x i8]* @__unique_stable_name.SC_Il12MyInt64ConstE3getEv, i64 0, i64 0) to i8 addrspace(4)*))
; CHECK-RT: %{{[0-9]+}} = call i64 @_Z20__spirv_SpecConstantix(i32 7, i64 0)
  %28 = tail call spir_func i64 @_Z33__sycl_getScalarSpecConstantValueImET_PKc(i8 addrspace(4)* addrspacecast (i8* getelementptr inbounds ([20 x i8], [20 x i8]* @__unique_stable_name.SC_Im13MyUInt64ConstE3getEv, i64 0, i64 0) to i8 addrspace(4)*))
; CHECK-RT: %{{[0-9]+}} = call i64 @_Z20__spirv_SpecConstantix(i32 8, i64 0)
  %29 = add i64 %28, %27
; CHECK-DEF: %[[SUM6:[0-9]+]] = add i64 0, 0
  %30 = add i64 %29, %26
; CHECK-DEF: %[[SUM7:[0-9]+]] = add i64 %[[SUM6]], %[[VAL5]]
  %31 = uitofp i64 %30 to float
; CHECK-DEF: %[[VAL6:[0-9]+]] = uitofp i64 %[[SUM7]] to float
  %32 = tail call spir_func float @_Z33__sycl_getScalarSpecConstantValueIfET_PKc(i8 addrspace(4)* addrspacecast (i8* getelementptr inbounds ([19 x i8], [19 x i8]* @__unique_stable_name.SC_If12MyFloatConstE3getEv, i64 0, i64 0) to i8 addrspace(4)*))
; CHECK-RT: %{{[0-9]+}} = call float @_Z20__spirv_SpecConstantif(i32 9, float 0.000000e+00)
  %33 = fadd float %32, %31
; CHECK-DEF: %[[SUM8:[0-9]+]] = fadd float 0.000000e+00, %[[VAL6]]
  %34 = fpext float %33 to double
; CHECK-DEF: %[[VAL7:[0-9]+]] = fpext float %[[SUM8]] to double
  %35 = tail call spir_func double @_Z33__sycl_getScalarSpecConstantValueIdET_PKc(i8 addrspace(4)* addrspacecast (i8* getelementptr inbounds ([20 x i8], [20 x i8]* @__unique_stable_name.SC_Id13MyDoubleConstE3getEv, i64 0, i64 0) to i8 addrspace(4)*))
; CHECK-RT: %{{[0-9]+}} = call double @_Z20__spirv_SpecConstantid(i32 10, double 0.000000e+00)
  %36 = fadd double %35, %34
; CHECK-DEF: %[[SUM9:[0-9]+]] = fadd double 0.000000e+00, %[[VAL7]]
  %37 = tail call spir_func double @_Z37__sycl_getScalar2020SpecConstantValueIdET_PKcPvS3_(i8 addrspace(4)* addrspacecast (i8* getelementptr inbounds ([21 x i8], [21 x i8]* @__unique_stable_name.SC_Id14MyDoubleConst2E3getEv, i64 0, i64 0) to i8 addrspace(4)*), i8 addrspace(4)* null, i8 addrspace(4)* null)
; CHECK-RT: %{{[0-9]+}} = call double @_Z20__spirv_SpecConstantid(i32 11, double 0.000000e+00)
  %38 = fadd double %37, %36
; CHECK-DEF: %[[GEP:[0-9a-z]+]] = getelementptr i8, i8 addrspace(4)* null, i32 0
; CHECK-DEF: %[[BITCAST:[0-9a-z]+]] = bitcast i8 addrspace(4)* %[[GEP]] to double addrspace(4)*
; CHECK-DEF: %[[LOAD:[0-9a-z]+]] = load double, double addrspace(4)* %[[BITCAST]], align 8
; CHECK-DEF: %[[SUM10:[0-9]+]] = fadd double %[[LOAD]], %[[SUM9]]
  %39 = fptrunc double %38 to float
; CHECK-DEF: %[[VAL8:[0-9]+]] = fptrunc double %[[SUM10]] to float
  %40 = addrspacecast float addrspace(1)* %7 to float addrspace(4)*
  store float %39, float addrspace(4)* %40, align 4, !tbaa !8
  ret void
}

declare dso_local spir_func zeroext i1 @_Z33__sycl_getScalarSpecConstantValueIbET_PKc(i8 addrspace(4)*) local_unnamed_addr #1

declare dso_local spir_func signext i8 @_Z33__sycl_getScalarSpecConstantValueIaET_PKc(i8 addrspace(4)*) local_unnamed_addr #1

declare dso_local spir_func zeroext i8 @_Z33__sycl_getScalarSpecConstantValueIhET_PKc(i8 addrspace(4)*) local_unnamed_addr #1

declare dso_local spir_func signext i16 @_Z33__sycl_getScalarSpecConstantValueIsET_PKc(i8 addrspace(4)*) local_unnamed_addr #1

declare dso_local spir_func zeroext i16 @_Z33__sycl_getScalarSpecConstantValueItET_PKc(i8 addrspace(4)*) local_unnamed_addr #1

declare dso_local spir_func i32 @_Z33__sycl_getScalarSpecConstantValueIiET_PKc(i8 addrspace(4)*) local_unnamed_addr #1

declare dso_local spir_func i32 @_Z33__sycl_getScalarSpecConstantValueIjET_PKc(i8 addrspace(4)*) local_unnamed_addr #1

declare dso_local spir_func i64 @_Z33__sycl_getScalarSpecConstantValueIlET_PKc(i8 addrspace(4)*) local_unnamed_addr #1

declare dso_local spir_func i64 @_Z33__sycl_getScalarSpecConstantValueImET_PKc(i8 addrspace(4)*) local_unnamed_addr #1

declare dso_local spir_func float @_Z33__sycl_getScalarSpecConstantValueIfET_PKc(i8 addrspace(4)*) local_unnamed_addr #1

declare dso_local spir_func double @_Z33__sycl_getScalarSpecConstantValueIdET_PKc(i8 addrspace(4)*) local_unnamed_addr #1

declare dso_local spir_func double @_Z37__sycl_getScalar2020SpecConstantValueIdET_PKcPvS3_(i8 addrspace(4)*, i8 addrspace(4)*, i8 addrspace(4)*) local_unnamed_addr #1

attributes #0 = { norecurse "correctly-rounded-divide-sqrt-fp-math"="false" "disable-tail-calls"="false" "frame-pointer"="all" "less-precise-fpmad"="false" "min-legal-vector-width"="0" "no-infs-fp-math"="false" "no-jump-tables"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "no-trapping-math"="false" "stack-protector-buffer-size"="8" "sycl-module-id"="/iusers/kbobrovs/ws/kbobrovs_llvm/sycl/test/spec_const/spec_const_types.cpp" "uniform-work-group-size"="true" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #1 = { "correctly-rounded-divide-sqrt-fp-math"="false" "disable-tail-calls"="false" "frame-pointer"="all" "less-precise-fpmad"="false" "no-infs-fp-math"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "no-trapping-math"="false" "stack-protector-buffer-size"="8" "unsafe-fp-math"="false" "use-soft-float"="false" }

!llvm.module.flags = !{!0}
!opencl.spir.version = !{!1}
!spirv.Source = !{!2}
!llvm.ident = !{!3}

!0 = !{i32 1, !"wchar_size", i32 4}
!1 = !{i32 1, i32 2}
!2 = !{i32 4, i32 100000}
!3 = !{!"clang version 11.0.0 (https://github.com/kbobrovs/llvm.git c9a794cf3cc06924bc0777f5facb507a98fad0a0)"}
!4 = !{i32 0, i32 0, i32 0, i32 0}
!5 = !{!"none", !"none", !"none", !"none"}
!6 = !{!"float*", !"cl::sycl::range<1>", !"cl::sycl::range<1>", !"cl::sycl::id<1>"}
!7 = !{!"", !"", !"", !""}
!8 = !{!9, !9, i64 0}
!9 = !{!"float", !10, i64 0}
!10 = !{!"omnipotent char", !11, i64 0}
!11 = !{!"Simple C++ TBAA"}

; CHECK-RT: !sycl.specialization-constants = !{![[#ID0:]], ![[#ID1:]], ![[#ID2:]], ![[#ID3:]], ![[#ID4:]], ![[#ID5:]], ![[#ID6:]], ![[#ID7:]], ![[#ID8:]], ![[#ID9:]], ![[#ID10:]], ![[#ID11:]]}

; CHECK-RT: ![[#ID0:]] = !{!"_ZTS11MyBoolConst", i32 0, i32 0, i32 1}
; CHECK-RT: ![[#ID1:]] = !{!"_ZTS11MyInt8Const", i32 1, i32 0, i32 1}
; CHECK-RT: ![[#ID2:]] = !{!"_ZTS12MyUInt8Const", i32 2, i32 0, i32 1}
; CHECK-RT: ![[#ID3:]] = !{!"_ZTS12MyInt16Const", i32 3, i32 0, i32 2}
; CHECK-RT: ![[#ID4:]] = !{!"_ZTS13MyUInt16Const", i32 4, i32 0, i32 2}
; CHECK-RT: ![[#ID5:]] = !{!"_ZTS12MyInt32Const", i32 5, i32 0, i32 4}
; CHECK-RT: ![[#ID6:]] = !{!"_ZTS13MyUInt32Const", i32 6, i32 0, i32 4}
; CHECK-RT: ![[#ID7:]] = !{!"_ZTS12MyInt64Const", i32 7, i32 0, i32 8}
; CHECK-RT: ![[#ID8:]] = !{!"_ZTS13MyUInt64Const", i32 8, i32 0, i32 8}
; CHECK-RT: ![[#ID9:]] = !{!"_ZTS12MyFloatConst", i32 9, i32 0, i32 4}
; CHECK-RT: ![[#ID10:]] = !{!"_ZTS13MyDoubleConst", i32 10, i32 0, i32 8}
; CHECK-RT: ![[#ID11:]] = !{!"_ZTS14MyDoubleConst2", i32 11, i32 0, i32 8}
