;; This tests diagnostic emission upon compile-time-unknown format string.
;; In this instance, a select instruction is given as the first argument
;; to __spirv_ocl_printf - as a result, moving the function into the constant AS
;; becomes impossible.

;; The IR is based on the following source/compilation (custom
;; build of SYCL Clang with SYCLMutatePrintfAddrspacePass turned off):
;; clang++ -fsycl -fsycl-device-only Inputs/experimental-printf-compile-time-unknown.cpp -S -D__SYCL_USE_NON_VARIADIC_SPIRV_OCL_PRINTF__

; RUN: not opt < %s --SYCLMutatePrintfAddrspace -S --enable-new-pm=0 2>&1 | FileCheck %s
; RUN: not opt < %s --passes=SYCLMutatePrintfAddrspace -S 2>&1 | FileCheck %s
; CHECK: error: experimental::printf requires format string to reside in constant address space. The compiler wasn't able to automatically convert your format string into constant address space when processing builtin _Z18__spirv_ocl_printf{{.*}} called in function {{.*}}foo{{.*}}.
; CHECK-NEXT: Make sure each format string literal is known at compile time or use OpenCL constant address space literals for device-side printf calls.

target datalayout = "e-i64:64-v16:16-v24:32-v32:32-v48:64-v96:128-v192:256-v256:256-v512:512-v1024:1024-n8:16:32:64"
target triple = "spir64-unknown-unknown"

%"class.cl::sycl::id" = type { %"class.cl::sycl::detail::array" }
%"class.cl::sycl::detail::array" = type { [1 x i64] }

$_ZTSZZ3fooiENKUlRN2cl4sycl7handlerEE_clES2_EUlvE_ = comdat any

@.str = private unnamed_addr addrspace(1) constant [10 x i8] c"String 0\0A\00", align 1
@.str.1 = private unnamed_addr addrspace(1) constant [10 x i8] c"String 1\0A\00", align 1

; Function Attrs: convergent norecurse
define weak_odr dso_local spir_kernel void @_ZTSZZ3fooiENKUlRN2cl4sycl7handlerEE_clES2_EUlvE_(i32 addrspace(1)* %_arg_, %"class.cl::sycl::id"* byval(%"class.cl::sycl::id") align 8 %_arg_3) local_unnamed_addr #0 comdat !kernel_arg_buffer_location !5 !sycl_kernel_omit_args !6 {
entry:
  %0 = getelementptr inbounds %"class.cl::sycl::id", %"class.cl::sycl::id"* %_arg_3, i64 0, i32 0, i32 0, i64 0
  %1 = addrspacecast i64* %0 to i64 addrspace(4)*
  %2 = load i64, i64 addrspace(4)* %1, align 8
  %add.ptr.i = getelementptr inbounds i32, i32 addrspace(1)* %_arg_, i64 %2
  %arrayidx.ascast.i.i = addrspacecast i32 addrspace(1)* %add.ptr.i to i32 addrspace(4)*
  %3 = load i32, i32 addrspace(4)* %arrayidx.ascast.i.i, align 4, !tbaa !7
  %cmp.i = icmp eq i32 %3, 0
  %..i = select i1 %cmp.i, i8 addrspace(4)* getelementptr inbounds ([10 x i8], [10 x i8] addrspace(4)* addrspacecast ([10 x i8] addrspace(1)* @.str to [10 x i8] addrspace(4)*), i64 0, i64 0), i8 addrspace(4)* getelementptr inbounds ([10 x i8], [10 x i8] addrspace(4)* addrspacecast ([10 x i8] addrspace(1)* @.str.1 to [10 x i8] addrspace(4)*), i64 0, i64 0)
  %call.i.i = tail call spir_func i32 @_Z18__spirv_ocl_printfIJEEiPKcDpT_(i8 addrspace(4)* %..i) #2
  ret void
}

; Function Attrs: convergent
declare dso_local spir_func i32 @_Z18__spirv_ocl_printfIJEEiPKcDpT_(i8 addrspace(4)*) local_unnamed_addr #1

attributes #0 = { convergent norecurse "frame-pointer"="all" "min-legal-vector-width"="0" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "sycl-module-id"="experimental-printf.cpp" "uniform-work-group-size"="true" }
attributes #1 = { convergent "frame-pointer"="all" "no-trapping-math"="true" "stack-protector-buffer-size"="8" }
attributes #2 = { convergent }

!llvm.module.flags = !{!0, !1}
!opencl.spir.version = !{!2}
!spirv.Source = !{!3}
!llvm.ident = !{!4}

!0 = !{i32 1, !"wchar_size", i32 4}
!1 = !{i32 7, !"frame-pointer", i32 2}
!2 = !{i32 1, i32 2}
!3 = !{i32 4, i32 100000}
!4 = !{!"clang version 14.0.0"}
!5 = !{i32 -1, i32 -1, i32 -1, i32 -1}
!6 = !{i1 false, i1 true, i1 true, i1 false}
!7 = !{!8, !8, i64 0}
!8 = !{!"int", !9, i64 0}
!9 = !{!"omnipotent char", !10, i64 0}
!10 = !{!"Simple C++ TBAA"}
