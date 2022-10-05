; Original code:
; #include <sycl/sycl.hpp>
; [[__sycl_detail__::__uses_aspects__(sycl::aspect::fp64, sycl::aspect::cpu)]] void foo() {}
; [[__sycl_detail__::__uses_aspects__(sycl::aspect::queue_profiling, sycl::aspect::cpu, sycl::aspect::image)]] void bar() {
;   foo();
; }
; int main() {
;   sycl::queue q;
;   q.submit([&](sycl::handler &cgh) {
;     cgh.single_task([=]() { bar(); });
;   });
; }

; RUN: sycl-post-link -split=auto %s -o %t.files.table
; RUN: FileCheck %s -input-file=%t.files_0.prop --check-prefix CHECK-PROP

; CHECK-PROP: [SYCL/device requirements]
; CHECK-PROP-NEXT: aspects=2|ACAAAAAAAAQAAAAAGAAAAkAAAAADAAAA

source_filename = "llvm-link"
target datalayout = "e-i64:64-v16:16-v24:32-v32:32-v48:64-v96:128-v192:256-v256:256-v512:512-v1024:1024-n8:16:32:64"
target triple = "spir64-unknown-unknown"

%class.anon = type { i8 }

$_ZTSZZ4mainENKUlRN4sycl3_V17handlerEE_clES2_EUlvE_ = comdat any

; Function Attrs: convergent mustprogress noinline norecurse optnone
define weak_odr dso_local spir_kernel void @_ZTSZZ4mainENKUlRN4sycl3_V17handlerEE_clES2_EUlvE_() #0 comdat !kernel_arg_buffer_location !6 {
entry:
  %__SYCLKernel = alloca %class.anon, align 1
  %__SYCLKernel.ascast = addrspacecast %class.anon* %__SYCLKernel to %class.anon addrspace(4)*
  call spir_func void @_ZZZ4mainENKUlRN4sycl3_V17handlerEE_clES2_ENKUlvE_clEv(%class.anon addrspace(4)* noundef align 1 dereferenceable_or_null(1) %__SYCLKernel.ascast) #3
  ret void
}

; Function Attrs: convergent mustprogress noinline norecurse optnone
define internal spir_func void @_ZZZ4mainENKUlRN4sycl3_V17handlerEE_clES2_ENKUlvE_clEv(%class.anon addrspace(4)* noundef align 1 dereferenceable_or_null(1) %this) #1 align 2 {
entry:
  %this.addr = alloca %class.anon addrspace(4)*, align 8
  %this.addr.ascast = addrspacecast %class.anon addrspace(4)** %this.addr to %class.anon addrspace(4)* addrspace(4)*
  store %class.anon addrspace(4)* %this, %class.anon addrspace(4)* addrspace(4)* %this.addr.ascast, align 8
  %this1 = load %class.anon addrspace(4)*, %class.anon addrspace(4)* addrspace(4)* %this.addr.ascast, align 8
  call spir_func void @_Z3barv() #3
  ret void
}

; Function Attrs: convergent mustprogress noinline norecurse optnone
define dso_local spir_func void @_Z3barv() #1 !intel_used_aspects !7 {
entry:
  call spir_func void @_Z3foov() #3
  ret void
}

; Function Attrs: convergent mustprogress noinline norecurse nounwind optnone
define dso_local spir_func void @_Z3foov() #2 !intel_used_aspects !8 {
entry:
  ret void
}

attributes #0 = { convergent mustprogress noinline norecurse optnone "frame-pointer"="all" "min-legal-vector-width"="0" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "sycl-module-id"="main.cpp" "uniform-work-group-size"="true" }
attributes #1 = { convergent mustprogress noinline norecurse optnone "frame-pointer"="all" "min-legal-vector-width"="0" "no-trapping-math"="true" "stack-protector-buffer-size"="8" }
attributes #2 = { convergent mustprogress noinline norecurse nounwind optnone "frame-pointer"="all" "min-legal-vector-width"="0" "no-trapping-math"="true" "stack-protector-buffer-size"="8" }
attributes #3 = { convergent }

!opencl.spir.version = !{!0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0}
!spirv.Source = !{!1, !1, !1, !1, !1, !1, !1, !1, !1, !1, !1, !1, !1, !1, !1, !1}
!llvm.ident = !{!2, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3}
!llvm.module.flags = !{!4, !5}

!0 = !{i32 1, i32 2}
!1 = !{i32 4, i32 100000}
!2 = !{!"clang version 16.0.0"}
!3 = !{!"clang version 16.0.0"}
!4 = !{i32 1, !"wchar_size", i32 4}
!5 = !{i32 7, !"frame-pointer", i32 2}
!6 = !{}
!7 = !{i32 12, i32 1, i32 9}
!8 = !{i32 6, i32 1}
