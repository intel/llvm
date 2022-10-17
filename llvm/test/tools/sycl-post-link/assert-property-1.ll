; This test checks that the post-link tool properly generates "assert used"
; property - it should include only kernels that call assertions in their call
; graph.

; RUN: sycl-post-link -split=auto -symbols -S %s -o %t.table
; RUN: FileCheck %s -input-file=%t_0.prop -check-prefix=PRESENCE-CHECK
; RUN: FileCheck %s -input-file=%t_0.prop -check-prefix=ABSENCE-CHECK

; SYCL source:
; void foo() {
;   assert(0);
; }
; void bar() {
;   assert(1);
; }
; void baz() {
;   foo();
; }
;
; int main() {
;   queue Q;
;   Q.submit([&] (handler& CGH) {
;     CGH.parallel_for<class TheKernel>(range<2>{2, 10}, [=](item<2> It) {
;       foo();
;     });
;     CGH.parallel_for<class TheKernel2>(range<2>{2, 10}, [=](item<2> It) {
;       bar();
;     });
;     CGH.parallel_for<class TheKernel3>(range<2>{2, 10}, [=](item<2> It) {
;       baz();
;       bar();
;     });
;   });
;   Q.wait();
;   return 0;
; }

target datalayout = "e-i64:64-v16:16-v24:32-v32:32-v48:64-v96:128-v192:256-v256:256-v512:512-v1024:1024-n8:16:32:64"
target triple = "spir64_x86_64-unknown-unknown"

%"class._ZTSN2cl4sycl5rangeILi2EEE.cl::sycl::range" = type { %"class._ZTSN2cl4sycl6detail5arrayILi2EEE.cl::sycl::detail::array" }
%"class._ZTSN2cl4sycl6detail5arrayILi2EEE.cl::sycl::detail::array" = type { [2 x i64] }
%"class._ZTSZZ4mainENK3$_0clERN2cl4sycl7handlerEEUlNS1_4itemILi2ELb1EEEE_.anon" = type { i8 }

@.str = private unnamed_addr addrspace(1) constant [2 x i8] c"0\00", align 1
@.str.1 = private unnamed_addr addrspace(1) constant [11 x i8] c"assert.cpp\00", align 1
@__PRETTY_FUNCTION__._Z3foov = private unnamed_addr addrspace(1) constant [11 x i8] c"void foo()\00", align 1
@__spirv_BuiltInGlobalInvocationId = external dso_local addrspace(1) constant <3 x i64>, align 32
@__spirv_BuiltInLocalInvocationId = external dso_local addrspace(1) constant <3 x i64>, align 32
@_ZL10assert_fmt = internal addrspace(2) constant [85 x i8] c"%s:%d: %s: global id: [%lu,%lu,%lu], local id: [%lu,%lu,%lu] Assertion `%s` failed.\0A\00", align 1

; PRESENCE-CHECK: [SYCL/assert used]

; Function Attrs: convergent norecurse nounwind mustprogress
define dso_local spir_func void @_Z3foov() {
entry:
  tail call spir_func void @__assert_fail(i8 addrspace(4)* getelementptr inbounds ([2 x i8], [2 x i8] addrspace(4)* addrspacecast ([2 x i8] addrspace(1)* @.str to [2 x i8] addrspace(4)*), i64 0, i64 0), i8 addrspace(4)* getelementptr inbounds ([11 x i8], [11 x i8] addrspace(4)* addrspacecast ([11 x i8] addrspace(1)* @.str.1 to [11 x i8] addrspace(4)*), i64 0, i64 0), i32 8, i8 addrspace(4)* getelementptr inbounds ([11 x i8], [11 x i8] addrspace(4)* addrspacecast ([11 x i8] addrspace(1)* @__PRETTY_FUNCTION__._Z3foov to [11 x i8] addrspace(4)*), i64 0, i64 0))
  ret void
}

; PRESENCE-CHECK-DAG: _ZTSZZ4mainENK3$_0clERN2cl4sycl7handlerEE9TheKernel
; Function Attrs: convergent norecurse
define weak_odr dso_local spir_kernel void @"_ZTSZZ4mainENK3$_0clERN2cl4sycl7handlerEE9TheKernel"() #0 {
entry:
  call spir_func void @_Z3foov()
  ret void
}

; Function Attrs: nofree norecurse nosync nounwind readnone willreturn mustprogress
define dso_local spir_func void @_Z3barv() {
entry:
  ret void
}

; ABSENCE-CHECK-NOT: _ZTSZZ4mainENK3$_0clERN2cl4sycl7handlerEE10TheKernel2
; Function Attrs: norecurse
define weak_odr dso_local spir_kernel void @"_ZTSZZ4mainENK3$_0clERN2cl4sycl7handlerEE10TheKernel2"() #1 {
entry:
  call spir_func void @_Z3barv()
  ret void
}

; Function Attrs: convergent inlinehint norecurse nounwind mustprogress
define internal spir_func void @"_ZZZ4mainENK3$_0clERN2cl4sycl7handlerEENKUlNS1_4itemILi2ELb1EEEE1_clES5_"() unnamed_addr #8 align 2 {
entry:
  call spir_func void @_Z3bazv()
  call spir_func void @_Z3barv()
  ret void
}

; Function Attrs: convergent norecurse nounwind mustprogress
define dso_local spir_func void @_Z3bazv() {
entry:
  tail call spir_func void @__assert_fail(i8 addrspace(4)* getelementptr inbounds ([2 x i8], [2 x i8] addrspace(4)* addrspacecast ([2 x i8] addrspace(1)* @.str to [2 x i8] addrspace(4)*), i64 0, i64 0), i8 addrspace(4)* getelementptr inbounds ([11 x i8], [11 x i8] addrspace(4)* addrspacecast ([11 x i8] addrspace(1)* @.str.1 to [11 x i8] addrspace(4)*), i64 0, i64 0), i32 8, i8 addrspace(4)* getelementptr inbounds ([11 x i8], [11 x i8] addrspace(4)* addrspacecast ([11 x i8] addrspace(1)* @__PRETTY_FUNCTION__._Z3foov to [11 x i8] addrspace(4)*), i64 0, i64 0))
  ret void
}

; PRESENCE-CHECK-DAG: _ZTSZZ4mainENK3$_0clERN2cl4sycl7handlerEE10TheKernel3
; Function Attrs: convergent norecurse
define weak_odr dso_local spir_kernel void @"_ZTSZZ4mainENK3$_0clERN2cl4sycl7handlerEE10TheKernel3"() #0 {
entry:
  call spir_func void @"_ZZZ4mainENK3$_0clERN2cl4sycl7handlerEENKUlNS1_4itemILi2ELb1EEEE1_clES5_"()
  ret void
}

; Function Attrs: convergent norecurse mustprogress
define weak dso_local spir_func void @__assert_fail(i8 addrspace(4)* %expr, i8 addrspace(4)* %file, i32 %line, i8 addrspace(4)* %func) {
entry:
  %call = tail call spir_func i64 @_Z28__spirv_GlobalInvocationId_xv()
  %call1 = tail call spir_func i64 @_Z28__spirv_GlobalInvocationId_yv()
  %call2 = tail call spir_func i64 @_Z28__spirv_GlobalInvocationId_zv()
  %call3 = tail call spir_func i64 @_Z27__spirv_LocalInvocationId_xv()
  %call4 = tail call spir_func i64 @_Z27__spirv_LocalInvocationId_yv()
  %call5 = tail call spir_func i64 @_Z27__spirv_LocalInvocationId_zv()
  tail call spir_func void @__devicelib_assert_fail(i8 addrspace(4)* %expr, i8 addrspace(4)* %file, i32 %line, i8 addrspace(4)* %func, i64 %call, i64 %call1, i64 %call2, i64 %call3, i64 %call4, i64 %call5)
  ret void
}

; Function Attrs: inlinehint norecurse mustprogress
declare dso_local spir_func i64 @_Z28__spirv_GlobalInvocationId_xv() local_unnamed_addr

; Function Attrs: inlinehint norecurse mustprogress
declare dso_local spir_func i64 @_Z28__spirv_GlobalInvocationId_yv() local_unnamed_addr

; Function Attrs: inlinehint norecurse mustprogress
declare dso_local spir_func i64 @_Z28__spirv_GlobalInvocationId_zv() local_unnamed_addr

; Function Attrs: inlinehint norecurse mustprogress
declare dso_local spir_func i64 @_Z27__spirv_LocalInvocationId_xv() local_unnamed_addr

; Function Attrs: inlinehint norecurse mustprogress
declare dso_local spir_func i64 @_Z27__spirv_LocalInvocationId_yv() local_unnamed_addr

; Function Attrs: inlinehint norecurse mustprogress
declare dso_local spir_func i64 @_Z27__spirv_LocalInvocationId_zv() local_unnamed_addr

; Function Attrs: convergent norecurse mustprogress
define weak dso_local spir_func void @__devicelib_assert_fail(i8 addrspace(4)* %expr, i8 addrspace(4)* %file, i32 %line, i8 addrspace(4)* %func, i64 %gid0, i64 %gid1, i64 %gid2, i64 %lid0, i64 %lid1, i64 %lid2) {
entry:
  %call = tail call spir_func i32 (i8 addrspace(2)*, ...) @_Z18__spirv_ocl_printfPU3AS2Kcz(i8 addrspace(2)* getelementptr inbounds ([85 x i8], [85 x i8] addrspace(2)* @_ZL10assert_fmt, i64 0, i64 0), i8 addrspace(4)* %file, i32 %line, i8 addrspace(4)* %func, i64 %gid0, i64 %gid1, i64 %gid2, i64 %lid0, i64 %lid1, i64 %lid2, i8 addrspace(4)* %expr)
  ret void
}

; Function Attrs: convergent
declare dso_local spir_func i32 @_Z18__spirv_ocl_printfPU3AS2Kcz(i8 addrspace(2)*, ...)

attributes #0 = { convergent norecurse "frame-pointer"="all" "min-legal-vector-width"="0" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "sycl-module-id"="assert.cpp" "uniform-work-group-size"="true" }
attributes #1 = { norecurse "frame-pointer"="all" "min-legal-vector-width"="0" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "sycl-module-id"="assert.cpp" "uniform-work-group-size"="true" }

!opencl.spir.version = !{!0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0}
!spirv.Source = !{!1, !1, !1, !1, !1, !1, !1, !1, !1, !1, !1}
!llvm.ident = !{!2, !2, !2, !2, !2, !2, !2, !2, !2, !2, !2}
!llvm.module.flags = !{!3, !4}

!0 = !{i32 1, i32 2}
!1 = !{i32 4, i32 100000}
!2 = !{!"clang version 13.0.0 (https://github.com/intel/llvm)"}
!3 = !{i32 1, !"wchar_size", i32 4}
!4 = !{i32 7, !"frame-pointer", i32 2}
!5 = !{i32 -1, i32 -1}
