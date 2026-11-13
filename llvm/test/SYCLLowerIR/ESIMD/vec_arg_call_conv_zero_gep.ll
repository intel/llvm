; RUN: opt -passes=esimd-opt-call-conv -S < %s | FileCheck %s

; Check that ESIMDOptimizeVecArgCallConvPass looks through an all-zero-index
; GEP when the actual parameter is the address of the first field of an alloca.

target datalayout = "e-i64:64-v16:16-v24:32-v32:32-v48:64-v96:128-v192:256-v256:256-v512:512-v1024:1024-n8:16:32:64"
target triple = "spir64-unknown-unknown"

%simd = type { %simd_impl }
%simd_impl = type { <16 x float> }

define dso_local spir_func void @callee(ptr addrspace(4) noalias sret(%simd) align 64 %agg.result, ptr noundef %x) #0 !sycl_explicit_simd !0 {
; CHECK-LABEL: define dso_local spir_func <16 x float> @callee(<16 x float> %{{.*}})
entry:
  %x.ascast = addrspacecast ptr %x to ptr addrspace(4)
  %v = load <16 x float>, ptr addrspace(4) %x.ascast, align 64
  store <16 x float> %v, ptr addrspace(4) %agg.result, align 64
  ret void
}

define dso_local spir_func void @caller(ptr addrspace(4) noalias sret(%simd) align 64 %agg.result, <16 x float> %in) #0 !sycl_explicit_simd !0 {
; CHECK-LABEL: define dso_local spir_func <16 x float> @caller(<16 x float> %{{.*}})
entry:
  %agg.tmp = alloca %simd, align 64
  %agg.tmp.ascast = addrspacecast ptr %agg.tmp to ptr addrspace(4)
  store <16 x float> %in, ptr addrspace(4) %agg.tmp.ascast, align 64
  %field = getelementptr inbounds %simd, ptr %agg.tmp, i64 0, i32 0
; CHECK:      %field = getelementptr inbounds %simd, ptr %agg.tmp, i64 0, i32 0
; CHECK-NEXT: [[VAL:%.*]] = load <16 x float>, ptr %field, align 64
; CHECK-NEXT: call spir_func <16 x float> @callee(<16 x float> [[VAL]])
  call spir_func void @callee(ptr addrspace(4) sret(%simd) align 64 %agg.result, ptr noundef nonnull %field)
  ret void
}

attributes #0 = { convergent noinline norecurse }
!0 = !{}
