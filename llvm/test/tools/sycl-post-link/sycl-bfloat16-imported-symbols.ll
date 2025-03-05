; This test checks that sycl-post-link can correctly analyze device code to
; detect bfloat16 devicelib functions are used and add these used functions
; too imported symbol list.

; RUN: sycl-post-link %s -emit-param-info -symbols -emit-imported-symbols -properties -split=auto -o %t.table
; RUN: FileCheck %s -input-file=%t_0.prop --check-prefix=CHECK-BF16

; CHECK-BF16: [SYCL/imported symbols]
; CHECK-BF16-NEXT: __devicelib_ConvertFToBF16INTEL

%"class.sycl::_V1::id" = type { %"class.sycl::_V1::detail::array" }
%"class.sycl::_V1::detail::array" = type { [1 x i64] }
%"class.sycl::_V1::ext::oneapi::bfloat16" = type { i16 }

$_ZTSZZ4mainENKUlRN4sycl3_V17handlerEE_clES2_EUlvE_ = comdat any

; Function Attrs: convergent mustprogress norecurse nounwind
define weak_odr dso_local spir_kernel void @_ZTSZZ4mainENKUlRN4sycl3_V17handlerEE_clES2_EUlvE_(ptr addrspace(1) noundef align 2 %_arg_bf16_acc, ptr noundef byval(%"class.sycl::_V1::id") align 8 %_arg_bf16_acc3, ptr addrspace(1) noundef readonly align 4 %_arg_fp32_acc, ptr noundef byval(%"class.sycl::_V1::id") align 8 %_arg_fp32_acc6) local_unnamed_addr #0 comdat !srcloc !6 !kernel_arg_buffer_location !7 !kernel_arg_runtime_aligned !8 !kernel_arg_exclusive_ptr !8 !sycl_fixed_targets !9 !sycl_kernel_omit_args !10 {
entry:
  %0 = load i64, ptr %_arg_bf16_acc3, align 8
  %add.ptr.i = getelementptr inbounds nuw %"class.sycl::_V1::ext::oneapi::bfloat16", ptr addrspace(1) %_arg_bf16_acc, i64 %0
  %1 = load i64, ptr %_arg_fp32_acc6, align 8
  %add.ptr.i25 = getelementptr inbounds nuw float, ptr addrspace(1) %_arg_fp32_acc, i64 %1
  %arrayidx.ascast.i = addrspacecast ptr addrspace(1) %add.ptr.i25 to ptr addrspace(4)
  %call.i.i = tail call spir_func noundef zeroext i16 @__devicelib_ConvertFToBF16INTEL(ptr addrspace(4) noundef align 4 dereferenceable(4) %arrayidx.ascast.i) #2
  store i16 %call.i.i, ptr addrspace(1) %add.ptr.i, align 2, !tbaa !11
  %arrayidx.i33 = getelementptr inbounds nuw i8, ptr addrspace(1) %add.ptr.i25, i64 4
  %arrayidx.ascast.i34 = addrspacecast ptr addrspace(1) %arrayidx.i33 to ptr addrspace(4)
  %call.i.i35 = tail call spir_func noundef zeroext i16 @__devicelib_ConvertFToBF16INTEL(ptr addrspace(4) noundef align 4 dereferenceable(4) %arrayidx.ascast.i34) #2
  %arrayidx.i38 = getelementptr inbounds nuw i8, ptr addrspace(1) %add.ptr.i, i64 2
  store i16 %call.i.i35, ptr addrspace(1) %arrayidx.i38, align 2, !tbaa !11
  %arrayidx.i42 = getelementptr inbounds nuw i8, ptr addrspace(1) %add.ptr.i25, i64 8
  %arrayidx.ascast.i43 = addrspacecast ptr addrspace(1) %arrayidx.i42 to ptr addrspace(4)
  %call.i.i44 = tail call spir_func noundef zeroext i16 @__devicelib_ConvertFToBF16INTEL(ptr addrspace(4) noundef align 4 dereferenceable(4) %arrayidx.ascast.i43) #2
  %arrayidx.i47 = getelementptr inbounds nuw i8, ptr addrspace(1) %add.ptr.i, i64 4
  store i16 %call.i.i44, ptr addrspace(1) %arrayidx.i47, align 2, !tbaa !11
  ret void
}

; Function Attrs: convergent nounwind
declare dso_local spir_func zeroext i16 @__devicelib_ConvertFToBF16INTEL(ptr addrspace(4) noundef align 4 dereferenceable(4)) local_unnamed_addr #1

declare dso_local spir_func i32 @_Z18__spirv_ocl_printfPU3AS2Kcz(ptr addrspace(2), ...)

attributes #0 = { convergent mustprogress norecurse nounwind "frame-pointer"="all" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "sycl-module-id"="bfloat16_post_tool.cpp" "sycl-optlevel"="2" "sycl-single-task" "uniform-work-group-size"="true" }
attributes #1 = { convergent nounwind "frame-pointer"="all" "no-trapping-math"="true" "stack-protector-buffer-size"="8" }
attributes #2 = { convergent nounwind }


!0 = !{i32 1, !"wchar_size", i32 4}
!1 = !{i32 1, !"sycl-device", i32 1}
!2 = !{i32 7, !"frame-pointer", i32 2}
!3 = !{i32 1, i32 2}
!4 = !{i32 4, i32 100000}
!5 = !{!"clang version 21.0.0git (https://github.com/jinge90/llvm.git 74a09d6598ef9bc735bf1fc2c7af09c9b43298e1)"}
!6 = !{i32 1593}
!7 = !{i32 -1, i32 -1, i32 -1, i32 -1}
!8 = !{i1 true, i1 false, i1 true, i1 false}
!9 = !{}
!10 = !{i1 false, i1 true, i1 true, i1 false, i1 false, i1 true, i1 true, i1 false}
!11 = !{!12, !12, i64 0}
!12 = !{!"short", !13, i64 0}
!13 = !{!"omnipotent char", !14, i64 0}
!14 = !{!"Simple C++ TBAA"}
