;; This tests replacement of string literal address space for __spirv_ocl_printf
;; at the regular O2 optimization level.

;; Compiled with the following command (custom build of SYCL Clang with
;; SYCLMutatePrintfAddrspacePass turned off):
;; clang++ -fsycl -fsycl-device-only experimental-printf.cpp -S -D__SYCL_USE_NON_VARIADIC_SPIRV_OCL_PRINTF__
;;
;; // experimental-printf.cpp
;; #include <CL/sycl.hpp>
;; using namespace sycl;
;; int main() {
;;   queue q;
;;   q.submit([&](handler &cgh) {
;;     cgh.single_task([=]() {
;;       ext::oneapi::experimental::printf("String No. %f\n", 1.0f);
;;       const char *IntFormatString = "String No. %i\n";
;;       ext::oneapi::experimental::printf(IntFormatString, 2);
;;       ext::oneapi::experimental::printf(IntFormatString, 3);
;;     });
;;   });
;;   return 0;
;; }

; RUN: opt < %s --SYCLMutatePrintfAddrspace -S | FileCheck %s
; RUN: opt < %s --SYCLMutatePrintfAddrspace -S --enable-new-pm=1 | FileCheck %s

target datalayout = "e-i64:64-v16:16-v24:32-v32:32-v48:64-v96:128-v192:256-v256:256-v512:512-v1024:1024-n8:16:32:64"
target triple = "spir64-unknown-unknown"

%"struct.cl::sycl::detail::AssertHappened" = type { i32, [257 x i8], [257 x i8], [129 x i8], i32, i64, i64, i64, i64, i64, i64 }
%"class.cl::sycl::range" = type { %"class.cl::sycl::detail::array" }
%"class.cl::sycl::detail::array" = type { [1 x i64] }
%"class.cl::sycl::id" = type { %"class.cl::sycl::detail::array" }

$_ZTSN2cl4sycl6detail23__sycl_service_kernel__16AssertInfoCopierE = comdat any

$_ZTSZZ4mainENKUlRN2cl4sycl7handlerEE_clES2_EUlvE_ = comdat any

; CHECK-DAG: @.str._AS2 = internal addrspace(2) constant [15 x i8] c"String No. %f\0A\00", align 1
@.str = private unnamed_addr addrspace(1) constant [15 x i8] c"String No. %f\0A\00", align 1
; CHECK-DAG: @.str.1._AS2 = internal addrspace(2) constant [15 x i8] c"String No. %i\0A\00", align 1
@.str.1 = private unnamed_addr addrspace(1) constant [15 x i8] c"String No. %i\0A\00", align 1

; Function Attrs: convergent norecurse
define weak_odr dso_local spir_kernel void @_ZTSN2cl4sycl6detail23__sycl_service_kernel__16AssertInfoCopierE(%"struct.cl::sycl::detail::AssertHappened" addrspace(1)* %_arg_, %"class.cl::sycl::range"* byval(%"class.cl::sycl::range") align 8 %_arg_1, %"class.cl::sycl::range"* byval(%"class.cl::sycl::range") align 8 %_arg_2, %"class.cl::sycl::id"* byval(%"class.cl::sycl::id") align 8 %_arg_3) local_unnamed_addr #0 comdat !kernel_arg_buffer_location !5 {
entry:
  %0 = getelementptr inbounds %"class.cl::sycl::id", %"class.cl::sycl::id"* %_arg_3, i64 0, i32 0, i32 0, i64 0
  %1 = addrspacecast i64* %0 to i64 addrspace(4)*
  %2 = load i64, i64 addrspace(4)* %1, align 8
  %add.ptr.i = getelementptr inbounds %"struct.cl::sycl::detail::AssertHappened", %"struct.cl::sycl::detail::AssertHappened" addrspace(1)* %_arg_, i64 %2
  %3 = bitcast %"struct.cl::sycl::detail::AssertHappened" addrspace(1)* %add.ptr.i to i8 addrspace(1)*
  %4 = addrspacecast i8 addrspace(1)* %3 to i8 addrspace(4)*
  tail call spir_func void @__devicelib_assert_read(i8 addrspace(4)* %4) #3
  ret void
}

; Function Attrs: convergent
declare extern_weak dso_local spir_func void @__devicelib_assert_read(i8 addrspace(4)*) local_unnamed_addr #1

; Function Attrs: convergent mustprogress norecurse
define weak_odr dso_local spir_kernel void @_ZTSZZ4mainENKUlRN2cl4sycl7handlerEE_clES2_EUlvE_() local_unnamed_addr #2 comdat !kernel_arg_buffer_location !6 {
entry:
  ; In particular, make sure that no argument promotion has been done for float
  ; upon variadic redeclaration:
  ; CHECK: tail call spir_func i32 (i8 addrspace(2)*, ...) @_Z18__spirv_ocl_printfPU3AS2Kcz(i8 addrspace(2)* getelementptr inbounds ([15 x i8], [15 x i8] addrspace(2)* @.str._AS2, i32 0, i32 0), float 1.000000e+00) #3
  %call.i.i = tail call spir_func i32 @_Z18__spirv_ocl_printfIJfEEiPKcDpT_(i8 addrspace(4)* getelementptr inbounds ([15 x i8], [15 x i8] addrspace(4)* addrspacecast ([15 x i8] addrspace(1)* @.str to [15 x i8] addrspace(4)*), i64 0, i64 0), float 1.000000e+00) #3
  ; CHECK-NEXT: tail call spir_func i32 (i8 addrspace(2)*, ...) @_Z18__spirv_ocl_printfPU3AS2Kcz(i8 addrspace(2)* getelementptr inbounds ([15 x i8], [15 x i8] addrspace(2)* @.str.1._AS2, i32 0, i32 0), i32 2) #3
  %call.i1.i = tail call spir_func i32 @_Z18__spirv_ocl_printfIJiEEiPKcDpT_(i8 addrspace(4)* getelementptr inbounds ([15 x i8], [15 x i8] addrspace(4)* addrspacecast ([15 x i8] addrspace(1)* @.str.1 to [15 x i8] addrspace(4)*), i64 0, i64 0), i32 2) #3
  ; CHECK-NEXT: tail call spir_func i32 (i8 addrspace(2)*, ...) @_Z18__spirv_ocl_printfPU3AS2Kcz(i8 addrspace(2)* getelementptr inbounds ([15 x i8], [15 x i8] addrspace(2)* @.str.1._AS2, i32 0, i32 0), i32 3) #3
  %call.i2.i = tail call spir_func i32 @_Z18__spirv_ocl_printfIJiEEiPKcDpT_(i8 addrspace(4)* getelementptr inbounds ([15 x i8], [15 x i8] addrspace(4)* addrspacecast ([15 x i8] addrspace(1)* @.str.1 to [15 x i8] addrspace(4)*), i64 0, i64 0), i32 3) #3
  ret void
}

; Make sure the non-variadic declarations have been wiped out
; in favor of the single variadic one:
; CHECK-NOT: declare dso_local spir_func i32 @_Z18__spirv_ocl_printf{{.*}}(i8 addrspace(4)*, float)
; CHECK-NOT: declare dso_local spir_func i32 @_Z18__spirv_ocl_printf{{.*}}(i8 addrspace(4)*, i32)
; CHECK: declare dso_local spir_func i32 @_Z18__spirv_ocl_printfPU3AS2Kcz(i8 addrspace(2)*, ...) #1

; Function Attrs: convergent
declare dso_local spir_func i32 @_Z18__spirv_ocl_printfIJfEEiPKcDpT_(i8 addrspace(4)*, float) local_unnamed_addr #1

; Function Attrs: convergent
declare dso_local spir_func i32 @_Z18__spirv_ocl_printfIJiEEiPKcDpT_(i8 addrspace(4)*, i32) local_unnamed_addr #1

attributes #0 = { convergent norecurse "frame-pointer"="all" "min-legal-vector-width"="0" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "sycl-module-id"="experimental-printf.cpp" "uniform-work-group-size"="true" }
attributes #1 = { convergent "frame-pointer"="all" "no-trapping-math"="true" "stack-protector-buffer-size"="8" }
attributes #2 = { convergent mustprogress norecurse "frame-pointer"="all" "min-legal-vector-width"="0" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "sycl-module-id"="experimental-printf.cpp" "uniform-work-group-size"="true" }
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
!6 = !{}
