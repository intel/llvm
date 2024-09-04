; Test checks handling of bool specialization constant.

; RUN: sycl-post-link -properties -split=auto -spec-const=native -S -o %t.table %s -generate-device-image-default-spec-consts
; RUN: FileCheck %s -input-file %t_1.ll --implicit-check-not="SpecConst"
; RUN: %if asserts %{ sycl-post-link -properties -debug-only=SpecConst -split=auto -spec-const=native -S %s -generate-device-image-default-spec-consts 2>&1 | FileCheck %s --check-prefix=CHECK-LOG %}

; CHECK: %bool1 = trunc i8 1 to i1
; CHECK: %frombool = zext i1 %bool1 to i8
; CHECK-LOG: sycl.specialization-constants
; CHECK-LOG:[[UNIQUE_PREFIX:[0-9a-zA-Z]+]]={0, 0, 1}
; CHECK-LOG: sycl.specialization-constants-default-values
; CHECK-LOG:{0, 1, 1}

target datalayout = "e-i64:64-v16:16-v24:32-v32:32-v48:64-v96:128-v192:256-v256:256-v512:512-v1024:1024-n8:16:32:64"
target triple = "spir64-unknown-unknown"

%"class.sycl::_V1::specialization_id" = type { i8 }
%struct.A = type { i8 }

@__usid_str = private unnamed_addr constant [28 x i8] c"uida046125e6e1c1f8d____ZL1c\00", align 1
@_ZL1c = internal addrspace(1) constant %"class.sycl::_V1::specialization_id" { i8 1 }, align 4

; Function Attrs: convergent nounwind
declare dso_local spir_func noundef zeroext i1 @_Z37__sycl_getScalar2020SpecConstantValueIbET_PKcPKvS4_(i8 addrspace(4)* noundef, i8 addrspace(4)* noundef, i8 addrspace(4)* noundef)

define spir_kernel void @kernel() {
entry:
  %bool = call spir_func noundef zeroext i1 @_Z37__sycl_getScalar2020SpecConstantValueIbET_PKcPKvS4_(i8 addrspace(4)* noundef addrspacecast (i8* getelementptr inbounds ([28 x i8], [28 x i8]* @__usid_str, i64 0, i64 0) to i8 addrspace(4)*), i8 addrspace(4)* noundef addrspacecast (i8 addrspace(1)* getelementptr inbounds (%"class.sycl::_V1::specialization_id", %"class.sycl::_V1::specialization_id" addrspace(1)* @_ZL1c, i64 0, i32 0) to i8 addrspace(4)*), i8 addrspace(4)* noundef null) #4
  %frombool = zext i1 %bool to i8
  ret void
}
