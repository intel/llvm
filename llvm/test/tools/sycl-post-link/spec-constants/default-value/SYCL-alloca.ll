; RUN: sycl-post-link -properties -split=auto -spec-const=native -S -o %t.table %s -generate-device-image-default-spec-consts
; RUN: FileCheck %s -input-file %t_1.ll --implicit-check-not="SpecConst"

; This test checks that the post link tool is able to correctly transform
; SYCL alloca intrinsics in SPIR-V devices when using default values.

%"class.sycl::_V1::specialization_id" = type { i64 }
%"class.sycl::_V1::specialization_id.0" = type { i32 }
%"class.sycl::_V1::specialization_id.1" = type { i16 }
%my_range = type { ptr addrspace(4), ptr addrspace(4) }

@size_i64 = internal addrspace(1) constant %"class.sycl::_V1::specialization_id" { i64 10 }, align 8
@size_i32 = internal addrspace(1) constant %"class.sycl::_V1::specialization_id.0" { i32 120 }, align 4
@size_i16 = internal addrspace(1) constant %"class.sycl::_V1::specialization_id.1" { i16 1 }, align 2

@size_i64_stable_name = private unnamed_addr constant [36 x i8] c"_ZTS14name_generatorIL_Z8size_i64EE\00", align 1
@size_i32_stable_name = private unnamed_addr constant [36 x i8] c"_ZTS14name_generatorIL_Z8size_i32EE\00", align 1
@size_i16_stable_name = private unnamed_addr constant [36 x i8] c"_ZTS14name_generatorIL_Z8size_i16EE\00", align 1

define dso_local void @private_alloca() {
; CHECK: alloca double, i32 120, align 8
  call ptr @llvm.sycl.alloca.p0.p4.p4.p4.f64(ptr addrspace(4) addrspacecast (ptr @size_i32_stable_name to ptr addrspace(4)), ptr addrspace(4) addrspacecast (ptr addrspace(1) @size_i32 to ptr addrspace(4)), ptr addrspace(4) null, double 0.000000e+00, i64 8)
; CHECK: alloca float, i64 10, align 8
  call ptr @llvm.sycl.alloca.p0.p4.p4.p4.f32(ptr addrspace(4) addrspacecast (ptr @size_i64_stable_name to ptr addrspace(4)), ptr addrspace(4) addrspacecast (ptr addrspace(1) @size_i64 to ptr addrspace(4)), ptr addrspace(4) null, float 0.000000e+00, i64 8)
; CHECK: alloca %my_range, i16 1, align 64
  call ptr @llvm.sycl.alloca.p0.p4.p4.p4.s_my_range(ptr addrspace(4) addrspacecast (ptr @size_i16_stable_name to ptr addrspace(4)), ptr addrspace(4) addrspacecast (ptr addrspace(1) @size_i16 to ptr addrspace(4)), ptr addrspace(4) null, %my_range zeroinitializer, i64 64)
  ret void
}

declare ptr @llvm.sycl.alloca.p0.p4.p4.p4.f32(ptr addrspace(4), ptr addrspace(4), ptr addrspace(4), float, i64)
declare ptr @llvm.sycl.alloca.p0.p4.p4.p4.f64(ptr addrspace(4), ptr addrspace(4), ptr addrspace(4), double, i64)
declare ptr @llvm.sycl.alloca.p0.p4.p4.p4.s_my_range(ptr addrspace(4), ptr addrspace(4), ptr addrspace(4), %my_range, i64)
