; RUN: sycl-post-link -spec-const=native < %s -S -o %t.table
; RUN: FileCheck %s -check-prefixes=CHECK-RT < %t_0.ll
; RUN: FileCheck %s --check-prefixes=CHECK-PROPS < %t_0.prop

; This test checks that the post link tool is able to correctly transform
; SYCL alloca intrinsics in SPIR-V devices.

%"class.sycl::_V1::specialization_id" = type { i64 }
%"class.sycl::_V1::specialization_id.0" = type { i32 }
%"class.sycl::_V1::specialization_id.1" = type { i16 }
%my_range = type { ptr addrspace(4), ptr addrspace(4) }

@size_i64 = internal addrspace(1) constant %"class.sycl::_V1::specialization_id" { i64 10 }, align 8
@size_i32 = internal addrspace(1) constant %"class.sycl::_V1::specialization_id.0" { i32 120 }, align 4
@size_i16 = internal addrspace(1) constant %"class.sycl::_V1::specialization_id.1" { i16 1 }, align 2

; Check that the following globals are preserved: even though they are not used
; in the module anymore, they could still be referenced by debug info metadata
; (specialization_id objects are used as template arguments in SYCL
; specialization constant APIs).
; CHECK: @size_i64
; CHECK: @size_i32
; CHECK: @size_i16

@size_i64_stable_name = private unnamed_addr constant [36 x i8] c"_ZTS14name_generatorIL_Z8size_i64EE\00", align 1
@size_i32_stable_name = private unnamed_addr constant [36 x i8] c"_ZTS14name_generatorIL_Z8size_i32EE\00", align 1
@size_i16_stable_name = private unnamed_addr constant [36 x i8] c"_ZTS14name_generatorIL_Z8size_i16EE\00", align 1

; CHECK-LABEL: define dso_local void @private_alloca
define dso_local void @private_alloca() {
; CHECK-RT: [[LENGTH:%.*]] = call i32 @_Z20__spirv_SpecConstantii(i32 1, i32 120)
; CHECK-RT: {{.*}} = alloca double, i32 [[LENGTH]], align 8
  call ptr @llvm.sycl.alloca.p0.p4.p4.p4.f64(ptr addrspace(4) addrspacecast (ptr @size_i32_stable_name to ptr addrspace(4)), ptr addrspace(4) addrspacecast (ptr addrspace(1) @size_i32 to ptr addrspace(4)), ptr addrspace(4) null, double 0.000000e+00, i64 8)
; CHECK-RT: [[LENGTH:%.*]] = call i64 @_Z20__spirv_SpecConstantix(i32 0, i64 10)
; CHECK-RT: {{.*}} = alloca float, i64 [[LENGTH]], align 8
  call ptr @llvm.sycl.alloca.p0.p4.p4.p4.f32(ptr addrspace(4) addrspacecast (ptr @size_i64_stable_name to ptr addrspace(4)), ptr addrspace(4) addrspacecast (ptr addrspace(1) @size_i64 to ptr addrspace(4)), ptr addrspace(4) null, float 0.000000e+00, i64 8)
  call ptr @llvm.sycl.alloca.p0.p4.p4.p4.s_my_range(ptr addrspace(4) addrspacecast (ptr @size_i16_stable_name to ptr addrspace(4)), ptr addrspace(4) addrspacecast (ptr addrspace(1) @size_i16 to ptr addrspace(4)), ptr addrspace(4) null, %my_range zeroinitializer, i64 64)
  ret void
}

declare ptr @llvm.sycl.alloca.p0.p4.p4.p4.f32(ptr addrspace(4), ptr addrspace(4), ptr addrspace(4), float, i64)
declare ptr @llvm.sycl.alloca.p0.p4.p4.p4.f64(ptr addrspace(4), ptr addrspace(4), ptr addrspace(4), double, i64)
declare ptr @llvm.sycl.alloca.p0.p4.p4.p4.s_my_range(ptr addrspace(4), ptr addrspace(4), ptr addrspace(4), %my_range, i64)

; CHECK-RT:  !sycl.specialization-constants = !{![[#ID0:]], ![[#ID1:]], ![[#ID2:]]}
; CHECK-RT:  !sycl.specialization-constants-default-values = !{![[#DEF0:]], ![[#DEF1:]], ![[#DEF2:]]}

; CHECK-RT:  ![[#ID0]] = !{!"_ZTS14name_generatorIL_Z8size_i64EE", i32 0, i32 0, i32 8}
; CHECK-RT:  ![[#ID1]] = !{!"_ZTS14name_generatorIL_Z8size_i32EE", i32 1, i32 0, i32 4}
; CHECK-RT:  ![[#ID2]] = !{!"_ZTS14name_generatorIL_Z8size_i16EE", i32 2, i32 0, i32 2}
; CHECK-RT:  ![[#DEF0]] = !{i64 10}
; CHECK-RT:  ![[#DEF1]] = !{i32 120}
; CHECK-RT:  ![[#DEF2]] = !{i16 1}

; CHECK-PROPS: [SYCL/specialization constants]
; CHECK-PROPS: _ZTS14name_generatorIL_Z8size_i64EE=2|
; CHECK-PROPS: _ZTS14name_generatorIL_Z8size_i32EE=2|
; CHECK-PROPS: _ZTS14name_generatorIL_Z8size_i16EE=2|
; CHECK-PROPS: [SYCL/specialization constants default values]
; CHECK-PROPS: all=2|
