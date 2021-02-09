; This test checks whether subroutine arguments are converted
; correctly to llvm's native vector type when callee is an extern function.
;
; RUN: opt < %s -ESIMDLowerVecArg -S | FileCheck %s

target datalayout = "e-i64:64-v16:16-v24:32-v32:32-v48:64-v96:128-v192:256-v256:256-v512:512-v1024:1024-n8:16:32:64"
target triple = "spir64-unknown-unknown-sycldevice"

%"class._ZTSN2cl4sycl5INTEL3gpu4simdIfLi16EEE.cl::sycl::INTEL::gpu::simd" = type { <16 x float> }

$"_ZTSZZ4mainENK3$_0clERN2cl4sycl7handlerEE4Test" = comdat any

@_ZL2VL = internal unnamed_addr addrspace(1) constant i32 16, align 4

; Function Attrs: convergent norecurse
define weak_odr dso_local spir_kernel void @"_ZTSZZ4mainENK3$_0clERN2cl4sycl7handlerEE4Test"() {
entry:
  %0 = alloca %"class._ZTSN2cl4sycl5INTEL3gpu4simdIfLi16EEE.cl::sycl::INTEL::gpu::simd", align 64
  %agg.tmp5.i = alloca %"class._ZTSN2cl4sycl5INTEL3gpu4simdIfLi16EEE.cl::sycl::INTEL::gpu::simd", align 64
  %agg.tmp6.i = alloca %"class._ZTSN2cl4sycl5INTEL3gpu4simdIfLi16EEE.cl::sycl::INTEL::gpu::simd", align 64
  %1 = addrspacecast %"class._ZTSN2cl4sycl5INTEL3gpu4simdIfLi16EEE.cl::sycl::INTEL::gpu::simd"* %0 to %"class._ZTSN2cl4sycl5INTEL3gpu4simdIfLi16EEE.cl::sycl::INTEL::gpu::simd" addrspace(4)*

; CHECK:  [[BITCASTRESULT1:%[a-zA-Z0-9_]*]] = bitcast {{.+}} addrspace(4)* %1 to <16 x float> addrspace(4)*
; CHECK:  [[BITCASTRESULT2:%[a-zA-Z0-9_]*]] = bitcast {{.+}} %agg.tmp5.i to <16 x float>*
; CHECK:  [[BITCASTRESULT3:%[a-zA-Z0-9_]*]] = bitcast {{.+}} %agg.tmp6.i to <16 x float>*
; CHECK-NEXT: call spir_func void @_Z4vaddN2cl4sycl5INTEL3gpu4simdIfLi16EEES4_(<16 x float> addrspace(4)* [[BITCASTRESULT1]], <16 x float>* [[BITCASTRESULT2]], <16 x float>* [[BITCASTRESULT3]])

  call spir_func void @_Z4vaddN2cl4sycl5INTEL3gpu4simdIfLi16EEES4_(%"class._ZTSN2cl4sycl5INTEL3gpu4simdIfLi16EEE.cl::sycl::INTEL::gpu::simd" addrspace(4)* sret(%"class._ZTSN2cl4sycl5INTEL3gpu4simdIfLi16EEE.cl::sycl::INTEL::gpu::simd") align 64 %1, %"class._ZTSN2cl4sycl5INTEL3gpu4simdIfLi16EEE.cl::sycl::INTEL::gpu::simd"* nonnull byval(%"class._ZTSN2cl4sycl5INTEL3gpu4simdIfLi16EEE.cl::sycl::INTEL::gpu::simd") align 64 %agg.tmp5.i, %"class._ZTSN2cl4sycl5INTEL3gpu4simdIfLi16EEE.cl::sycl::INTEL::gpu::simd"* nonnull byval(%"class._ZTSN2cl4sycl5INTEL3gpu4simdIfLi16EEE.cl::sycl::INTEL::gpu::simd") align 64 %agg.tmp6.i) #1
  ret void
}

; CHECK: declare dso_local spir_func void @_Z4vaddN2cl4sycl5INTEL3gpu4simdIfLi16EEES4_(<16 x float> addrspace(4)*, <16 x float>*, <16 x float>*){{.+}}
; Function Attrs: convergent
declare dso_local spir_func void @_Z4vaddN2cl4sycl5INTEL3gpu4simdIfLi16EEES4_(%"class._ZTSN2cl4sycl5INTEL3gpu4simdIfLi16EEE.cl::sycl::INTEL::gpu::simd" addrspace(4)* sret(%"class._ZTSN2cl4sycl5INTEL3gpu4simdIfLi16EEE.cl::sycl::INTEL::gpu::simd") align 64 %0, %"class._ZTSN2cl4sycl5INTEL3gpu4simdIfLi16EEE.cl::sycl::INTEL::gpu::simd"* byval(%"class._ZTSN2cl4sycl5INTEL3gpu4simdIfLi16EEE.cl::sycl::INTEL::gpu::simd") align 64 %1, %"class._ZTSN2cl4sycl5INTEL3gpu4simdIfLi16EEE.cl::sycl::INTEL::gpu::simd"* byval(%"class._ZTSN2cl4sycl5INTEL3gpu4simdIfLi16EEE.cl::sycl::INTEL::gpu::simd") align 64 %2) local_unnamed_addr #2

attributes #0 = { convergent "disable-tail-calls"="false" "frame-pointer"="all" "less-precise-fpmad"="false" "no-infs-fp-math"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #1 = { convergent }


