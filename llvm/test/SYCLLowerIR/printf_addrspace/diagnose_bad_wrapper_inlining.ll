;; This tests diagnostic emission whenever printf's format string argument
;; is taken from the argument list of a user wrapper function, but then the
;; user-provided wrapper does not get inlined (while the library-provided
;; experimental::printf wrapper does).

;; The IR for test purposes is based on the following source/compilation (custom
;; build of SYCL Clang with SYCLMutatePrintfAddrspacePass turned off):
;; clang++ -fsycl -fsycl-device-only Inputs/experimental-printf-bad-inline-test.cpp -S -D__SYCL_USE_NON_VARIADIC_SPIRV_OCL_PRINTF__ -O2

; RUN: not opt < %s --SYCLMutatePrintfAddrspace -S --enable-new-pm=0 2>&1 | FileCheck %s
; RUN: not opt < %s --passes=SYCLMutatePrintfAddrspace -S 2>&1 | FileCheck %s
; CHECK: error: experimental::printf requires format string to reside in constant address space. The compiler wasn't able to automatically convert your format string into constant address space when processing builtin _Z18__spirv_ocl_printf{{.*}} called in function {{.*}}custom_wrapper{{.*}}.
; CHECK-NEXT: Consider simplifying the code by passing format strings directly into experimental::printf calls, avoiding indirection via wrapper function arguments.

target datalayout = "e-i64:64-v16:16-v24:32-v32:32-v48:64-v96:128-v192:256-v256:256-v512:512-v1024:1024-n8:16:32:64"
target triple = "spir64-unknown-unknown"

%"class.cl::sycl::id" = type { %"class.cl::sycl::detail::array" }
%"class.cl::sycl::detail::array" = type { [1 x i64] }

; Function Attrs: convergent mustprogress norecurse
define dso_local spir_func void @_Z14custom_wrapperPKc(i8 addrspace(4)* %S) local_unnamed_addr #0 {
entry:
  %call.i = tail call spir_func i32 @_Z18__spirv_ocl_printfIJEEiPKcDpT_(i8 addrspace(4)* %S) #3
  ret void
}

; Function Attrs: convergent
declare dso_local spir_func i32 @_Z18__spirv_ocl_printfIJEEiPKcDpT_(i8 addrspace(4)*) local_unnamed_addr #1

attributes #0 = { convergent mustprogress norecurse "frame-pointer"="all" "min-legal-vector-width"="0" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "sycl-module-id"="experimental-printf.cpp" }
attributes #1 = { convergent "frame-pointer"="all" "no-trapping-math"="true" "stack-protector-buffer-size"="8" }
attributes #2 = { convergent norecurse "frame-pointer"="all" "min-legal-vector-width"="0" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "sycl-module-id"="experimental-printf-bad-inline-test.cpp" "uniform-work-group-size"="true" }
attributes #3 = { convergent }

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
