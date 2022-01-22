;; This tests diagnostic emission whenever printf's format string argument
;; is taken from the argument list of a user wrapper function and no
;; inlining is performed prior to the address space mutation pass.

;; The IR for test purposes is based on the following source/compilation (custom
;; build of SYCL Clang with SYCLMutatePrintfAddrspacePass turned off):
;; clang++ -fsycl -fsycl-device-only Inputs/experimental-printf-bad-inline-test.cpp -S -D__SYCL_USE_NON_VARIADIC_SPIRV_OCL_PRINTF__ -O0

; RUN: not opt < %s --SYCLMutatePrintfAddrspace -S --enable-new-pm=0 2>&1 | FileCheck %s
; RUN: not opt < %s --passes=SYCLMutatePrintfAddrspace -S 2>&1 | FileCheck %s
; CHECK: error: experimental::printf requires format string to reside in constant address space. The compiler wasn't able to automatically convert your format string into constant address space when processing builtin _ZN2cl4sycl3ext6oneapi12experimental6printf{{.*}} called in function {{.*}}custom_wrapper{{.*}}.
; CHECK-NEXT: Consider simplifying the code by passing format strings directly into experimental::printf calls, avoiding indirection via wrapper function arguments.

target datalayout = "e-i64:64-v16:16-v24:32-v32:32-v48:64-v96:128-v192:256-v256:256-v512:512-v1024:1024-n8:16:32:64"
target triple = "spir64-unknown-unknown"

%"class.cl::sycl::range" = type { %"class.cl::sycl::detail::array" }
%"class.cl::sycl::detail::array" = type { [1 x i64] }
%"class.cl::sycl::id" = type { %"class.cl::sycl::detail::array" }
%class.anon = type { %"class.cl::sycl::accessor" }
%"class.cl::sycl::accessor" = type { %"class.cl::sycl::detail::AccessorImplDevice" }
%"class.cl::sycl::detail::AccessorImplDevice" = type { %"class.cl::sycl::id", %"class.cl::sycl::range", %"class.cl::sycl::range" }
%"class.cl::sycl::detail::accessor_common" = type { i8 }

$_ZN2cl4sycl3ext6oneapi12experimental6printfIcJEEEiPKT_DpT0_ = comdat any

; Function Attrs: convergent mustprogress noinline norecurse optnone
define dso_local spir_func void @_Z14custom_wrapperPKc(i8 addrspace(4)* %S) #0 {
entry:
  %S.addr = alloca i8 addrspace(4)*, align 8
  %S.addr.ascast = addrspacecast i8 addrspace(4)** %S.addr to i8 addrspace(4)* addrspace(4)*
  store i8 addrspace(4)* %S, i8 addrspace(4)* addrspace(4)* %S.addr.ascast, align 8
  %0 = load i8 addrspace(4)*, i8 addrspace(4)* addrspace(4)* %S.addr.ascast, align 8
  %call = call spir_func i32 @_ZN2cl4sycl3ext6oneapi12experimental6printfIcJEEEiPKT_DpT0_(i8 addrspace(4)* %0) #9
  ret void
}

; Function Attrs: convergent mustprogress noinline norecurse optnone
define linkonce_odr dso_local spir_func i32 @_ZN2cl4sycl3ext6oneapi12experimental6printfIcJEEEiPKT_DpT0_(i8 addrspace(4)* %__format) #1 comdat {
entry:
  %retval = alloca i32, align 4
  %__format.addr = alloca i8 addrspace(4)*, align 8
  %retval.ascast = addrspacecast i32* %retval to i32 addrspace(4)*
  %__format.addr.ascast = addrspacecast i8 addrspace(4)** %__format.addr to i8 addrspace(4)* addrspace(4)*
  store i8 addrspace(4)* %__format, i8 addrspace(4)* addrspace(4)* %__format.addr.ascast, align 8
  %0 = load i8 addrspace(4)*, i8 addrspace(4)* addrspace(4)* %__format.addr.ascast, align 8
  %call = call spir_func i32 @_Z18__spirv_ocl_printfIJEEiPKcDpT_(i8 addrspace(4)* %0) #9
  ret i32 %call
}

; Function Attrs: convergent
declare dso_local spir_func i32 @_Z18__spirv_ocl_printfIJEEiPKcDpT_(i8 addrspace(4)*) #2

attributes #0 = { convergent mustprogress noinline norecurse optnone "frame-pointer"="all" "min-legal-vector-width"="0" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "sycl-module-id"="experimental-printf-bad-inline-test.cpp" }
attributes #1 = { convergent mustprogress noinline norecurse optnone "frame-pointer"="all" "min-legal-vector-width"="0" "no-trapping-math"="true" "stack-protector-buffer-size"="8" }
attributes #2 = { convergent "frame-pointer"="all" "no-trapping-math"="true" "stack-protector-buffer-size"="8" }
attributes #3 = { convergent mustprogress noinline norecurse optnone "frame-pointer"="all" "min-legal-vector-width"="0" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "sycl-module-id"="experimental-printf-bad-inline-test.cpp" "uniform-work-group-size"="true" }
attributes #4 = { convergent noinline norecurse optnone "frame-pointer"="all" "min-legal-vector-width"="0" "no-trapping-math"="true" "stack-protector-buffer-size"="8" }
attributes #5 = { argmemonly nofree nounwind willreturn }
attributes #6 = { argmemonly nofree nounwind willreturn writeonly }
attributes #7 = { convergent noinline norecurse nounwind optnone "frame-pointer"="all" "min-legal-vector-width"="0" "no-trapping-math"="true" "stack-protector-buffer-size"="8" }
attributes #8 = { convergent mustprogress noinline norecurse nounwind optnone "frame-pointer"="all" "min-legal-vector-width"="0" "no-trapping-math"="true" "stack-protector-buffer-size"="8" }
attributes #9 = { convergent }

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
!6 = distinct !{!6, !7, !8}
!7 = !{!"llvm.loop.mustprogress"}
!8 = !{!"llvm.loop.unroll.enable"}
