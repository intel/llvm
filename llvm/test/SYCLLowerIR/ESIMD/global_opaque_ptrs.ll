; This test checks whether globals are converted
; correctly to llvm's native vector type with opaque pointers.
;
; RUN: opt -opaque-pointers < %s -passes=LowerESIMD -S | FileCheck %s

target datalayout = "e-i64:64-v16:16-v24:32-v32:32-v48:64-v96:128-v192:256-v256:256-v512:512-v1024:1024-n8:16:32:64"
target triple = "spir64-unknown-unknown"

%"class.sycl::_V1::ext::intel::esimd::simd" = type { %"class.sycl::_V1::ext::intel::esimd::detail::simd_obj_impl" }
%"class.sycl::_V1::ext::intel::esimd::detail::simd_obj_impl" = type { <16 x float> }

; CHECK: [[NEWGLOBAL:[@a-zA-Z0-9_]*]] = dso_local global <16 x float> zeroinitializer, align 64 #[[#Attr:]]
@va = dso_local global %"class.sycl::_V1::ext::intel::esimd::simd" zeroinitializer, align 64 #0

define weak_odr dso_local spir_kernel void @foo() !sycl_explicit_simd !0 {
entry:
; CHECK: call <16 x float> @llvm.genx.vload.v16f32.p4(ptr addrspace(4) addrspacecast (ptr [[NEWGLOBAL]] to ptr addrspace(4)))
  %0 = call spir_func noundef <16 x float> @_Z13__esimd_vloadIfLi16EEN4sycl3_V13ext5intel5esimd6detail15raw_vector_typeIT_XT0_EE4typeEPKS9_(ptr addrspace(4) noundef addrspacecast (ptr @va to ptr addrspace(4)))
  ret void
}

; Function Attrs: convergent nounwind
declare dso_local spir_func noundef <16 x float> @_Z13__esimd_vloadIfLi16EEN4sycl3_V13ext5intel5esimd6detail15raw_vector_typeIT_XT0_EE4typeEPKS9_(ptr addrspace(4) noundef) local_unnamed_addr

attributes #0 = { "genx_byte_offset"="192" "genx_volatile" }
!0 = !{}