; This test checks whether global stores are converted to vstores
;
; RUN: opt < %s -passes=LowerESIMD -S | FileCheck %s                            

target datalayout = "e-i64:64-v16:16-v24:32-v32:32-v48:64-v96:128-v192:256-v256:256-v512:512-v1024:1024-n8:16:32:64"
target triple = "spir64-unknown-unknown"

%"class.sycl::_V1::ext::intel::esimd::simd" = type { %"class.sycl::_V1::ext::intel::esimd::detail::simd_obj_impl" }
%"class.sycl::_V1::ext::intel::esimd::detail::simd_obj_impl" = type { <16 x float> }

@va = dso_local global %"class.sycl::_V1::ext::intel::esimd::simd" zeroinitializer, align 64 #0
@vb = dso_local global %"class.sycl::_V1::ext::intel::esimd::simd" zeroinitializer, align 64 #0

define weak_odr dso_local spir_kernel void @foo() #1 {
%1 = call <16 x float> asm "", "=rw"()
; CHECK: call void @llvm.genx.vstore.v16f32.p0(<16 x float> %1, ptr @va)
store <16 x float> %1, ptr @va
; CHECK-NEXT: @llvm.genx.vstore.v16f32.p0(<16 x float> %1, ptr @vb)
store <16 x float> %1, ptr @vb
ret void
}

attributes #0 = { "genx_byte_offset"="0" "genx_volatile" }
; CHECK-NOT: noinline
attributes #1 = { noinline }
