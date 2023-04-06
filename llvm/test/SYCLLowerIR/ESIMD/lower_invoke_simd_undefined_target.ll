; RUN: opt -passes=lower-invoke-simd -S < %s | FileCheck %s
; This test checks we add VCStackCall to the simd helper
; even if it is undefined.
target datalayout = "e-i64:64-v16:16-v24:32-v32:32-v48:64-v96:128-v192:256-v256:256-v512:512-v1024:1024-n8:16:32:64"
target triple = "spir64-unknown-unknown"
declare x86_regcallcc noundef float @_Z33__regcall3____builtin_invoke_simdXX(<16 x float> (<16 x float> (<16 x float>, <16 x float>)*, <16 x float>, <16 x float>)* noundef, <16 x float> (<16 x float>, <16 x float>)* noundef, float noundef, float noundef) local_unnamed_addr #1


; CHECK: define {{.*}} @SIMD_CALL_HELPER{{.*}} #[[HELPER_ATTRS:[0-9]+]]
; CHECK: #[[HELPER_ATTRS]] = { {{.*}} "VCStackCall"
define linkonce_odr dso_local x86_regcallcc <16 x float> @SIMD_CALL_HELPER (<16 x float> (<16 x float>, <16 x float>)* noundef %f, <16 x float> %simd_args.coerce, <16 x float> %simd_args.coerce3) #2 {
entry:
  %call = tail call x86_regcallcc <16 x float> %f(<16 x float> %simd_args.coerce, <16 x float> %simd_args.coerce3) #3
  ret <16 x float> %call
}

define dso_local x86_regcallcc void @__regcall3__foo(float noundef %in_buffer, float addrspace(4)* %out_buffer.coerce, i32 noundef %index, <16 x float> (<16 x float>, <16 x float>)* noundef %callback) local_unnamed_addr #0 !sycl_explicit_simd !6 {
entry:
  %conv = sitofp i32 %index to float
  %call4.i = tail call x86_regcallcc noundef float @_Z33__regcall3____builtin_invoke_simdXX(<16 x float> (<16 x float> (<16 x float>, <16 x float>)*, <16 x float>, <16 x float>)* noundef nonnull @SIMD_CALL_HELPER, <16 x float> (<16 x float>, <16 x float>)* noundef %callback, float noundef %in_buffer, float noundef %conv) #3
  %idxprom = sext i32 %index to i64
  %arrayidx = getelementptr inbounds float, float addrspace(4)* %out_buffer.coerce, i64 %idxprom
  store float %call4.i, float addrspace(4)* %arrayidx, align 4
  ret void
}

attributes #0 = { convergent mustprogress noinline norecurse "frame-pointer"="all" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "sycl-module-id"="test.cpp" }
attributes #1 = { convergent "frame-pointer"="all" "no-trapping-math"="true" "stack-protector-buffer-size"="8" }
attributes #2 = { convergent mustprogress norecurse "frame-pointer"="all" "no-trapping-math"="true" "referenced-indirectly" "stack-protector-buffer-size"="8" "sycl-module-id"="test.cpp" }
attributes #3 = { convergent }

!llvm.module.flags = !{!0, !1, !2, !3}
!opencl.spir.version = !{!4}
!spirv.Source = !{!5}

!0 = !{i32 1, !"wchar_size", i32 4}
!1 = !{i32 8, !"PIC Level", i32 2}
!2 = !{i32 7, !"PIE Level", i32 2}
!3 = !{i32 7, !"frame-pointer", i32 2}
!4 = !{i32 1, i32 2}
!5 = !{i32 4, i32 100000}
!6 = !{}
