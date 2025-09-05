; RUN: opt -passes=spec-constants %s -S -o - | FileCheck %s

; Check there is no assert error when literal address is loaded from an alloca
; with offset.

target datalayout = "e-i64:64-v16:16-v24:32-v32:32-v48:64-v96:128-v192:256-v256:256-v512:512-v1024:1024-n8:16:32:64-G1"
target triple = "spir64-unknown-unknown"

%"class.sycl::_V1::specialization_id" = type { i32 }

@_ZL9test_id_1 = addrspace(1) constant %"class.sycl::_V1::specialization_id" { i32 42 }
@__usid_str = constant [36 x i8] c"uide7faddc6b4d2fe92____ZL9test_id_1\00"

define spir_func void @_ZZZ4mainENKUlRN4sycl3_V17handlerEE_clES2_ENKUlNS0_14kernel_handlerEE_clES4_(ptr addrspace(4) %this1.i7) {
entry:
  %MyAlloca = alloca i8, i64 224, align 32
  %0 = ptrtoint ptr %MyAlloca to i64
  %1 = add i64 %0, 96
  %2 = inttoptr i64 %1 to ptr
  %SymbolicID.ascast.i = addrspacecast ptr %2 to ptr addrspace(4)
  store ptr addrspace(4) addrspacecast (ptr @__usid_str to ptr addrspace(4)), ptr addrspace(4) %SymbolicID.ascast.i, align 8
  %3 = load ptr addrspace(4), ptr addrspace(4) %SymbolicID.ascast.i, align 8
  %4 = load ptr addrspace(4), ptr addrspace(4) %this1.i7, align 8

; CHECK-NOT: call spir_func noundef i32 @_Z37__sycl_getScalar2020SpecConstantValueIiET_PKcPKvS4_(
; CHECK: %conv = sitofp i32 %load to double

  %call.i8 = call spir_func i32 @_Z37__sycl_getScalar2020SpecConstantValueIiET_PKcPKvS4_(ptr addrspace(4) %3, ptr addrspace(4) addrspacecast (ptr addrspace(1) @_ZL9test_id_1 to ptr addrspace(4)), ptr addrspace(4) %4)
  %conv = sitofp i32 %call.i8 to double
  ret void
}

declare spir_func i32 @_Z37__sycl_getScalar2020SpecConstantValueIiET_PKcPKvS4_(ptr addrspace(4), ptr addrspace(4), ptr addrspace(4))
