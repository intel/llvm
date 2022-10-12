; ModuleID = '/localdisk2/etiotto/intel-llvm/polygeist/tools/cgeist/Test/Verification/sycl/single_task.cpp'
source_filename = "/localdisk2/etiotto/intel-llvm/polygeist/tools/cgeist/Test/Verification/sycl/single_task.cpp"
target datalayout = "e-i64:64-v16:16-v24:32-v32:32-v48:64-v96:128-v192:256-v256:256-v512:512-v1024:1024-n8:16:32:64"
target triple = "spir64-unknown-unknown"

%"class.sycl::_V1::id" = type { %"class.sycl::_V1::detail::array" }
%"class.sycl::_V1::detail::array" = type { [1 x i64] }

$_ZTSZZ16host_single_taskvENKUlRN4sycl3_V17handlerEE_clES2_E18kernel_single_task = comdat any

; Function Attrs: norecurse
define weak_odr dso_local spir_kernel void @_ZTSZZ16host_single_taskvENKUlRN4sycl3_V17handlerEE_clES2_E18kernel_single_task(i32 addrspace(1)* noundef align 4 %_arg_A, %"class.sycl::_V1::id"* noundef byval(%"class.sycl::_V1::id") align 8 %_arg_A3) local_unnamed_addr #0 comdat !kernel_arg_buffer_location !5 !kernel_arg_runtime_aligned !6 !kernel_arg_exclusive_ptr !6 !sycl_kernel_omit_args !7 {
entry:
  call void @__itt_offload_wi_start_wrapper()
  %0 = getelementptr inbounds %"class.sycl::_V1::id", %"class.sycl::_V1::id"* %_arg_A3, i64 0, i32 0, i32 0, i64 0
  %1 = addrspacecast i64* %0 to i64 addrspace(4)*
  %2 = load i64, i64 addrspace(4)* %1, align 8
  %add.ptr.i = getelementptr inbounds i32, i32 addrspace(1)* %_arg_A, i64 %2
  %arrayidx.ascast.i.i = addrspacecast i32 addrspace(1)* %add.ptr.i to i32 addrspace(4)*
  store i32 42, i32 addrspace(4)* %arrayidx.ascast.i.i, align 4, !tbaa !8
  call void @__itt_offload_wi_finish_wrapper()
  ret void
}

declare dso_local spir_func i32 @_Z18__spirv_ocl_printfPU3AS2Kcz(i8 addrspace(2)*, ...)

declare void @__itt_offload_wi_start_wrapper()

declare void @__itt_offload_wi_finish_wrapper()

attributes #0 = { norecurse "frame-pointer"="all" "min-legal-vector-width"="0" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "sycl-module-id"="/localdisk2/etiotto/intel-llvm/polygeist/tools/cgeist/Test/Verification/sycl/single_task.cpp" "uniform-work-group-size"="true" }

!llvm.module.flags = !{!0, !1}
!opencl.spir.version = !{!2}
!spirv.Source = !{!3}
!llvm.ident = !{!4}

!0 = !{i32 1, !"wchar_size", i32 4}
!1 = !{i32 7, !"frame-pointer", i32 2}
!2 = !{i32 1, i32 2}
!3 = !{i32 4, i32 100000}
!4 = !{!"clang version 16.0.0 (https://github.com/etiotto/intel-llvm.git 08dc010ec6efac3dcc50f4ab3d478058fbeee855)"}
!5 = !{i32 -1, i32 -1, i32 -1, i32 -1}
!6 = !{i1 true, i1 false, i1 false, i1 false}
!7 = !{i1 false, i1 true, i1 true, i1 false}
!8 = !{!9, !9, i64 0}
!9 = !{!"int", !10, i64 0}
!10 = !{!"omnipotent char", !11, i64 0}
!11 = !{!"Simple C++ TBAA"}
