;; This tests replacement of string literal address space for the variadic version of
;; __spirv_ocl_printf when no optimizations (inlining, constant propagation) have been
;; performed prior to the pass scheduling.
;; Note: this test's checks are identical to those for the non-variadic version
;; of pre-transformation printf functions in non-optimized mode.

;; Compiled with the following command (custom build of SYCL Clang with
;; SYCLMutatePrintfAddrspacePass turned off):
;; clang++ -fsycl -fsycl-device-only Inputs/experimental-printf.cpp -S -O0

; RUN: opt < %s --SYCLMutatePrintfAddrspace -S | FileCheck %s
; RUN: opt < %s --SYCLMutatePrintfAddrspace -S --enable-new-pm=1 | FileCheck %s

target datalayout = "e-i64:64-v16:16-v24:32-v32:32-v48:64-v96:128-v192:256-v256:256-v512:512-v1024:1024-n8:16:32:64"
target triple = "spir64-unknown-unknown"

%"class.cl::sycl::range" = type { %"class.cl::sycl::detail::array" }
%"class.cl::sycl::detail::array" = type { [1 x i64] }
%"class.cl::sycl::id" = type { %"class.cl::sycl::detail::array" }
%class.anon = type { %"class.cl::sycl::accessor" }
%"class.cl::sycl::accessor" = type { %"class.cl::sycl::detail::AccessorImplDevice" }
%"class.cl::sycl::detail::AccessorImplDevice" = type { %"class.cl::sycl::id", %"class.cl::sycl::range", %"class.cl::sycl::range" }
%"class.cl::sycl::detail::accessor_common" = type { i8 }
%class.anon.0 = type { i8 }

$_ZTSZZ4mainENKUlRN2cl4sycl7handlerEE_clES2_EUlvE_ = comdat any

$_ZN2cl4sycl3ext6oneapi12experimental6printfIcJfEEEiPKT_DpT0_ = comdat any

$_ZN2cl4sycl3ext6oneapi12experimental6printfIcJiEEEiPKT_DpT0_ = comdat any

; CHECK-DAG: @.str._AS2 = internal addrspace(2) constant [15 x i8] c"String No. %f\0A\00", align 1
@.str = private unnamed_addr addrspace(1) constant [15 x i8] c"String No. %f\0A\00", align 1
; CHECK-DAG: @.str.1._AS2 = internal addrspace(2) constant [15 x i8] c"String No. %i\0A\00", align 1
@.str.1 = private unnamed_addr addrspace(1) constant [15 x i8] c"String No. %i\0A\00", align 1

; Function Attrs: convergent mustprogress noinline norecurse optnone
define weak_odr dso_local spir_kernel void @_ZTSZZ4mainENKUlRN2cl4sycl7handlerEE_clES2_EUlvE_() #0 comdat !kernel_arg_buffer_location !9 {
entry:
  %0 = alloca %class.anon.0, align 1
  %1 = addrspacecast %class.anon.0* %0 to %class.anon.0 addrspace(4)*
  call spir_func void @_ZZZ4mainENKUlRN2cl4sycl7handlerEE_clES2_ENKUlvE_clEv(%class.anon.0 addrspace(4)* align 1 dereferenceable_or_null(1) %1) #8
  ret void
}

; CHECK-LABEL: define internal spir_func void @_ZZZ4main{{.*}}
; Function Attrs: convergent mustprogress noinline norecurse optnone
define internal spir_func void @_ZZZ4mainENKUlRN2cl4sycl7handlerEE_clES2_ENKUlvE_clEv(%class.anon.0 addrspace(4)* align 1 dereferenceable_or_null(1) %this) #2 align 2 {
entry:
  %this.addr = alloca %class.anon.0 addrspace(4)*, align 8
  %IntFormatString = alloca i8 addrspace(4)*, align 8
  %this.addr.ascast = addrspacecast %class.anon.0 addrspace(4)** %this.addr to %class.anon.0 addrspace(4)* addrspace(4)*
  %IntFormatString.ascast = addrspacecast i8 addrspace(4)** %IntFormatString to i8 addrspace(4)* addrspace(4)*
  store %class.anon.0 addrspace(4)* %this, %class.anon.0 addrspace(4)* addrspace(4)* %this.addr.ascast, align 8
  %this1 = load %class.anon.0 addrspace(4)*, %class.anon.0 addrspace(4)* addrspace(4)* %this.addr.ascast, align 8
  ; In particular, make sure that no argument promotion has been done for float
  ; upon variadic redeclaration:
  ; CHECK: call spir_func i32 (i8 addrspace(2)*, ...) @_Z18__spirv_ocl_printfPU3AS2Kcz(i8 addrspace(2)* getelementptr inbounds ([15 x i8], [15 x i8] addrspace(2)* @.str._AS2, i32 0, i32 0), float 1.000000e+00)
  %call = call spir_func i32 @_ZN2cl4sycl3ext6oneapi12experimental6printfIcJfEEEiPKT_DpT0_(i8 addrspace(4)* getelementptr inbounds ([15 x i8], [15 x i8] addrspace(4)* addrspacecast ([15 x i8] addrspace(1)* @.str to [15 x i8] addrspace(4)*), i64 0, i64 0), float 1.000000e+00) #8
  store i8 addrspace(4)* getelementptr inbounds ([15 x i8], [15 x i8] addrspace(4)* addrspacecast ([15 x i8] addrspace(1)* @.str.1 to [15 x i8] addrspace(4)*), i64 0, i64 0), i8 addrspace(4)* addrspace(4)* %IntFormatString.ascast, align 8
  %0 = load i8 addrspace(4)*, i8 addrspace(4)* addrspace(4)* %IntFormatString.ascast, align 8
  ; CHECK: call spir_func i32 (i8 addrspace(2)*, ...) @_Z18__spirv_ocl_printfPU3AS2Kcz(i8 addrspace(2)* getelementptr inbounds ([15 x i8], [15 x i8] addrspace(2)* @.str.1._AS2, i32 0, i32 0), i32 2)
  %call2 = call spir_func i32 @_ZN2cl4sycl3ext6oneapi12experimental6printfIcJiEEEiPKT_DpT0_(i8 addrspace(4)* %0, i32 2) #8
  %1 = load i8 addrspace(4)*, i8 addrspace(4)* addrspace(4)* %IntFormatString.ascast, align 8
  ; CHECK: call spir_func i32 (i8 addrspace(2)*, ...) @_Z18__spirv_ocl_printfPU3AS2Kcz(i8 addrspace(2)* getelementptr inbounds ([15 x i8], [15 x i8] addrspace(2)* @.str.1._AS2, i32 0, i32 0), i32 3)
  %call3 = call spir_func i32 @_ZN2cl4sycl3ext6oneapi12experimental6printfIcJiEEEiPKT_DpT0_(i8 addrspace(4)* %1, i32 3) #8
  ret void
}

; CHECK-LABEL: declare dso_local spir_func i32 @_Z18__spirv_ocl_printfPU3AS2Kcz(i8 addrspace(2)*, ...)

; Function Attrs: convergent mustprogress noinline norecurse optnone
define linkonce_odr dso_local spir_func i32 @_ZN2cl4sycl3ext6oneapi12experimental6printfIcJfEEEiPKT_DpT0_(i8 addrspace(4)* %__format, float %args) #2 comdat {
entry:
  %retval = alloca i32, align 4
  %__format.addr = alloca i8 addrspace(4)*, align 8
  %args.addr = alloca float, align 4
  %retval.ascast = addrspacecast i32* %retval to i32 addrspace(4)*
  %__format.addr.ascast = addrspacecast i8 addrspace(4)** %__format.addr to i8 addrspace(4)* addrspace(4)*
  %args.addr.ascast = addrspacecast float* %args.addr to float addrspace(4)*
  store i8 addrspace(4)* %__format, i8 addrspace(4)* addrspace(4)* %__format.addr.ascast, align 8
  store float %args, float addrspace(4)* %args.addr.ascast, align 4
  %0 = load i8 addrspace(4)*, i8 addrspace(4)* addrspace(4)* %__format.addr.ascast, align 8
  %1 = load float, float addrspace(4)* %args.addr.ascast, align 4
  %call = call spir_func i32 (i8 addrspace(4)*, ...) @_Z18__spirv_ocl_printfPKcz(i8 addrspace(4)* %0, float %1) #8
  ret i32 %call
}

; Function Attrs: convergent mustprogress noinline norecurse optnone
define linkonce_odr dso_local spir_func i32 @_ZN2cl4sycl3ext6oneapi12experimental6printfIcJiEEEiPKT_DpT0_(i8 addrspace(4)* %__format, i32 %args) #2 comdat {
entry:
  %retval = alloca i32, align 4
  %__format.addr = alloca i8 addrspace(4)*, align 8
  %args.addr = alloca i32, align 4
  %retval.ascast = addrspacecast i32* %retval to i32 addrspace(4)*
  %__format.addr.ascast = addrspacecast i8 addrspace(4)** %__format.addr to i8 addrspace(4)* addrspace(4)*
  %args.addr.ascast = addrspacecast i32* %args.addr to i32 addrspace(4)*
  store i8 addrspace(4)* %__format, i8 addrspace(4)* addrspace(4)* %__format.addr.ascast, align 8
  store i32 %args, i32 addrspace(4)* %args.addr.ascast, align 4
  %0 = load i8 addrspace(4)*, i8 addrspace(4)* addrspace(4)* %__format.addr.ascast, align 8
  %1 = load i32, i32 addrspace(4)* %args.addr.ascast, align 4
  %call = call spir_func i32 (i8 addrspace(4)*, ...) @_Z18__spirv_ocl_printfPKcz(i8 addrspace(4)* %0, i32 %1) #8
  ret i32 %call
}

; Function Attrs: convergent
declare dso_local spir_func i32 @_Z18__spirv_ocl_printfPKcz(i8 addrspace(4)*, ...) #7

attributes #0 = { convergent mustprogress noinline norecurse optnone "frame-pointer"="all" "min-legal-vector-width"="0" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "sycl-module-id"="experimental-printf.cpp" "uniform-work-group-size"="true" }
attributes #1 = { convergent noinline norecurse optnone "frame-pointer"="all" "min-legal-vector-width"="0" "no-trapping-math"="true" "stack-protector-buffer-size"="8" }
attributes #2 = { convergent mustprogress noinline norecurse optnone "frame-pointer"="all" "min-legal-vector-width"="0" "no-trapping-math"="true" "stack-protector-buffer-size"="8" }
attributes #3 = { argmemonly nofree nounwind willreturn }
attributes #4 = { argmemonly nofree nounwind willreturn writeonly }
attributes #5 = { convergent noinline norecurse nounwind optnone "frame-pointer"="all" "min-legal-vector-width"="0" "no-trapping-math"="true" "stack-protector-buffer-size"="8" }
attributes #6 = { convergent mustprogress noinline norecurse nounwind optnone "frame-pointer"="all" "min-legal-vector-width"="0" "no-trapping-math"="true" "stack-protector-buffer-size"="8" }
attributes #7 = { convergent "frame-pointer"="all" "no-trapping-math"="true" "stack-protector-buffer-size"="8" }
attributes #8 = { convergent }

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
!9 = !{}
