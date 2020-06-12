; RUN: opt -ESIMDLowerLoadStore -S < %s | FileCheck %s

%"class._ZTSN4sycl5intel3gpu4simdIiLi16EEE.sycl::intel::gpu::simd" = type { <16 x i32> }

@vg = dso_local global %"class._ZTSN4sycl5intel3gpu4simdIiLi16EEE.sycl::intel::gpu::simd" zeroinitializer, align 64 #0
@vc = dso_local addrspace(1) global <16 x i32> zeroinitializer, align 64

; Function Attrs: norecurse nounwind
define dso_local spir_func void @_Z3foov() local_unnamed_addr #1 {
; CHECK-LABEL: @_Z3foov(
; CHECK-NEXT: [[TMP1:%.*]] = call <16 x i32> @llvm.genx.vload.v16i32.p0v16i32(<16 x i32>* getelementptr inbounds (%"class._ZTSN4sycl5intel3gpu4simdIiLi16EEE.sycl::intel::gpu::simd", %"class._ZTSN4sycl5intel3gpu4simdIiLi16EEE.sycl::intel::gpu::simd"* @vg, i64 0, i32 0))
; CHECK-NEXT: store <16 x i32> [[TMP1]], <16 x i32> addrspace(4)* addrspacecast (<16 x i32> addrspace(1)* @vc to <16 x i32> addrspace(4)*), align 64

  %call.esimd = call <16 x i32> @llvm.genx.vload.v16i32.p4v16i32(<16 x i32> addrspace(4)* addrspacecast (<16 x i32>* getelementptr inbounds (%"class._ZTSN4sycl5intel3gpu4simdIiLi16EEE.sycl::intel::gpu::simd", %"class._ZTSN4sycl5intel3gpu4simdIiLi16EEE.sycl::intel::gpu::simd"* @vg, i64 0, i32 0) to <16 x i32> addrspace(4)*))
  call void @llvm.genx.vstore.v16i32.p4v16i32(<16 x i32> %call.esimd, <16 x i32> addrspace(4)* addrspacecast (<16 x i32> addrspace(1)* @vc to <16 x i32> addrspace(4)*))
  ret void
}

; Function Attrs: nounwind
declare !genx_intrinsic_id !1 <16 x i32> @llvm.genx.vload.v16i32.p4v16i32(<16 x i32> addrspace(4)*) #2

; Function Attrs: nounwind
declare !genx_intrinsic_id !2 void @llvm.genx.vstore.v16i32.p4v16i32(<16 x i32>, <16 x i32> addrspace(4)*) #2

attributes #0 = { "genx_byte_offset"="192" "genx_volatile" }
attributes #1 = { norecurse nounwind "correctly-rounded-divide-sqrt-fp-math"="false" "disable-tail-calls"="false" "frame-pointer"="none" "less-precise-fpmad"="false" "min-legal-vector-width"="512" "no-infs-fp-math"="false" "no-jump-tables"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "no-trapping-math"="false" "stack-protector-buffer-size"="8" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #2 = { nounwind }

!0 = !{}
!1 = !{i32 8269}
!2 = !{i32 8270}

