;; This tests replacement of string literal address space for the variadic version of
;; __spirv_ocl_printf at the regular O2 optimization level.
;; Note: this test's checks are almost identical to those for non-variadic version
;; of pre-transformation printf functions. However, we can't exclude argument promotion
;; here since it has been enforced by FE.

;; Compiled with the following command (custom build of SYCL Clang with
;; SYCLMutatePrintfAddrspacePass turned off):
;; clang++ -fsycl -fsycl-device-only Inputs/experimental-printf.cpp -S -O2

; RUN: opt < %s --SYCLMutatePrintfAddrspace -S | FileCheck %s
; RUN: opt < %s --SYCLMutatePrintfAddrspace -S --enable-new-pm=1 | FileCheck %s

target datalayout = "e-i64:64-v16:16-v24:32-v32:32-v48:64-v96:128-v192:256-v256:256-v512:512-v1024:1024-n8:16:32:64"
target triple = "spir64-unknown-unknown"

%"class.cl::sycl::range" = type { %"class.cl::sycl::detail::array" }
%"class.cl::sycl::detail::array" = type { [1 x i64] }
%"class.cl::sycl::id" = type { %"class.cl::sycl::detail::array" }

$_ZTSZZ4mainENKUlRN2cl4sycl7handlerEE_clES2_EUlvE_ = comdat any

; CHECK-DAG: @.str._AS2 = internal addrspace(2) constant [15 x i8] c"String No. %f\0A\00", align 1
@.str = private unnamed_addr addrspace(1) constant [15 x i8] c"String No. %f\0A\00", align 1
; CHECK-DAG: @.str.1._AS2 = internal addrspace(2) constant [15 x i8] c"String No. %i\0A\00", align 1
@.str.1 = private unnamed_addr addrspace(1) constant [15 x i8] c"String No. %i\0A\00", align 1

; Function Attrs: convergent mustprogress norecurse
define weak_odr dso_local spir_kernel void @_ZTSZZ4mainENKUlRN2cl4sycl7handlerEE_clES2_EUlvE_() local_unnamed_addr #2 comdat !kernel_arg_buffer_location !6 {
entry:
  ; CHECK: tail call spir_func i32 (i8 addrspace(2)*, ...) @_Z18__spirv_ocl_printfPU3AS2Kcz(i8 addrspace(2)* getelementptr inbounds ([15 x i8], [15 x i8] addrspace(2)* @.str._AS2, i32 0, i32 0), double 1.000000e+00)
  %call.i.i = tail call spir_func i32 (i8 addrspace(4)*, ...) @_Z18__spirv_ocl_printfPKcz(i8 addrspace(4)* getelementptr inbounds ([15 x i8], [15 x i8] addrspace(4)* addrspacecast ([15 x i8] addrspace(1)* @.str to [15 x i8] addrspace(4)*), i64 0, i64 0), double 1.000000e+00) #3
  ; CHECK-NEXT: tail call spir_func i32 (i8 addrspace(2)*, ...) @_Z18__spirv_ocl_printfPU3AS2Kcz(i8 addrspace(2)* getelementptr inbounds ([15 x i8], [15 x i8] addrspace(2)* @.str.1._AS2, i32 0, i32 0), i32 2)
  %call.i1.i = tail call spir_func i32 (i8 addrspace(4)*, ...) @_Z18__spirv_ocl_printfPKcz(i8 addrspace(4)* getelementptr inbounds ([15 x i8], [15 x i8] addrspace(4)* addrspacecast ([15 x i8] addrspace(1)* @.str.1 to [15 x i8] addrspace(4)*), i64 0, i64 0), i32 2) #3
  ; CHECK-NEXT: tail call spir_func i32 (i8 addrspace(2)*, ...) @_Z18__spirv_ocl_printfPU3AS2Kcz(i8 addrspace(2)* getelementptr inbounds ([15 x i8], [15 x i8] addrspace(2)* @.str.1._AS2, i32 0, i32 0), i32 3)
  %call.i2.i = tail call spir_func i32 (i8 addrspace(4)*, ...) @_Z18__spirv_ocl_printfPKcz(i8 addrspace(4)* getelementptr inbounds ([15 x i8], [15 x i8] addrspace(4)* addrspacecast ([15 x i8] addrspace(1)* @.str.1 to [15 x i8] addrspace(4)*), i64 0, i64 0), i32 3) #3
  ret void
}

; CHECK: declare dso_local spir_func i32 @_Z18__spirv_ocl_printfPU3AS2Kcz(i8 addrspace(2)*, ...)
; Function Attrs: convergent
declare dso_local spir_func i32 @_Z18__spirv_ocl_printfPKcz(i8 addrspace(4)*, ...) local_unnamed_addr #1

attributes #0 = { convergent norecurse "frame-pointer"="all" "min-legal-vector-width"="0" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "sycl-module-id"="../../../tests/experimental-printf.cpp" "uniform-work-group-size"="true" }
attributes #1 = { convergent "frame-pointer"="all" "no-trapping-math"="true" "stack-protector-buffer-size"="8" }
attributes #2 = { convergent mustprogress norecurse "frame-pointer"="all" "min-legal-vector-width"="0" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "sycl-module-id"="../../../tests/experimental-printf.cpp" "uniform-work-group-size"="true" }
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
