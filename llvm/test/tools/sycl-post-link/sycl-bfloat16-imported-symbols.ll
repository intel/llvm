; This test checks that sycl-post-link can correctly analyze device code to
; detect bfloat16 devicelib functions are used and add these used functions
; too imported symbol list.

; RUN: sycl-post-link %s -emit-param-info -symbols -emit-imported-symbols -properties -split=auto -o %t.table
; RUN: FileCheck %s -input-file=%t_0.prop --check-prefix=CHECK-BF16

; CHECK-BF16: [SYCL/imported symbols]
; CHECK-BF16-NEXT: __devicelib_ConvertFToBF16INTEL

%"class.sycl::_V1::ext::oneapi::bfloat16" = type { i16 }

$_Z17__sycl_kernel_foov = comdat any

define weak_odr dso_local spir_kernel void @_Z17__sycl_kernel_foov(ptr addrspace(1) align 2 %_arg_bf16_acc, ptr align 8 %_arg_bf16_acc3, ptr addrspace(1) align 4 %_arg_fp32_acc, ptr align 8 %_arg_fp32_acc6) local_unnamed_addr #0 {
entry:
  %0 = load i64, ptr %_arg_bf16_acc3, align 8
  %1 = load i64, ptr %_arg_fp32_acc6, align 8
  %add.ptr.i25 = getelementptr inbounds nuw float, ptr addrspace(1) %_arg_fp32_acc, i64 %1
  %arrayidx.ascast.i = addrspacecast ptr addrspace(1) %add.ptr.i25 to ptr addrspace(4)
  %call.i.i = tail call spir_func noundef zeroext i16 @__devicelib_ConvertFToBF16INTEL(ptr addrspace(4) noundef align 4 dereferenceable(4) %arrayidx.ascast.i) #1
  ret void
}

; Function Attrs: convergent nounwind
declare dso_local spir_func zeroext i16 @__devicelib_ConvertFToBF16INTEL(ptr addrspace(4) noundef align 4 dereferenceable(4)) local_unnamed_addr #0

attributes #0 = { convergent nounwind "frame-pointer"="all" "no-trapping-math"="true" "stack-protector-buffer-size"="8" }
attributes #1 = { convergent nounwind }
