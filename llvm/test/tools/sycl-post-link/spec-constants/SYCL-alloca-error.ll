; RUN: sycl-post-link -properties -spec-const=emulation %s 2>&1 | FileCheck %s

; This test checks the `-spec-const` pass on SPIR-V targets and emulation mode,
; i.e., on AOT SPIR-V targets. In this scenario, 'llvm.sycl.alloca' intrinsics
; must be left unmodified.

; Note that coming from clang this case should never be reached.

; CHECK: sycl-post-link NOTE: no modifications to the input LLVM IR have been made

target triple = "spir64_x86_64"

%"class.sycl::_V1::specialization_id" = type { i64 }

@size_i64 = addrspace(1) constant %"class.sycl::_V1::specialization_id" { i64 10 }, align 8

@size_i64_stable_name = private unnamed_addr constant [36 x i8] c"_ZTS14name_generatorIL_Z8size_i64EE\00", align 1

define dso_local void @private_alloca() {
  call ptr @llvm.sycl.alloca.p0.p4.p4.p4.f64(ptr addrspace(4) addrspacecast (ptr @size_i64_stable_name to ptr addrspace(4)), ptr addrspace(4) addrspacecast (ptr addrspace(1) @size_i64 to ptr addrspace(4)), ptr addrspace(4) null, double 0.000000e+00, i64 8)
  ret void
}

declare ptr @llvm.sycl.alloca.p0.p4.p4.p4.f32(ptr addrspace(4), ptr addrspace(4), ptr addrspace(4), float, i64)
