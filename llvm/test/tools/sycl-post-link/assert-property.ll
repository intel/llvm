; This test checks that the post-link tool properly generates "assert used"
; property - it should include only kernels that call assertions in their call
; graph.

; RUN: sycl-post-link -split=auto -symbols -S %s -o %t.table
; RUN: FileCheck %s -input-file=%t_0.prop

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
;       bar();
;       baz();
;     });
;   });
;   Q.wait();
;   return 0;
; }

target datalayout = "e-i64:64-v16:16-v24:32-v32:32-v48:64-v96:128-v192:256-v256:256-v512:512-v1024:1024-n8:16:32:64"
target triple = "spir64_x86_64-unknown-unknown-sycldevice"

%"class._ZTSN2cl4sycl5rangeILi2EEE.cl::sycl::range" = type { %"class._ZTSN2cl4sycl6detail5arrayILi2EEE.cl::sycl::detail::array" }
%"class._ZTSN2cl4sycl6detail5arrayILi2EEE.cl::sycl::detail::array" = type { [2 x i64] }
%"class._ZTSZZ4mainENK3$_0clERN2cl4sycl7handlerEEUlNS1_4itemILi2ELb1EEEE_.anon" = type { i8 }

@.str = private unnamed_addr addrspace(1) constant [2 x i8] c"0\00", align 1
@.str.1 = private unnamed_addr addrspace(1) constant [11 x i8] c"assert.cpp\00", align 1
@__PRETTY_FUNCTION__._Z3foov = private unnamed_addr addrspace(1) constant [11 x i8] c"void foo()\00", align 1
@__spirv_BuiltInGlobalInvocationId = external dso_local addrspace(1) constant <3 x i64>, align 32
@__spirv_BuiltInLocalInvocationId = external dso_local addrspace(1) constant <3 x i64>, align 32
@_ZL10assert_fmt = internal addrspace(2) constant [85 x i8] c"%s:%d: %s: global id: [%lu,%lu,%lu], local id: [%lu,%lu,%lu] Assertion `%s` failed.\0A\00", align 1

; CHECK: [SYCL/assert used]

; CHECK: _ZTSN2cl4sycl6detail19__pf_kernel_wrapperIZZ4mainENK3$_0clERNS0_7handlerEE9TeKernelEE
; Function Attrs: convergent norecurse
define weak_odr dso_local spir_kernel void @"_ZTSN2cl4sycl6detail19__pf_kernel_wrapperIZZ4mainENK3$_0clERNS0_7handlerEE9TeKernelEE"(%"class._ZTSN2cl4sycl5rangeILi2EEE.cl::sycl::range"* byval(%"class._ZTSN2cl4sycl5rangeILi2EEE.cl::sycl::range") align 8 %_arg_, %"class._ZTSZZ4mainENK3$_0clERN2cl4sycl7handlerEEUlNS1_4itemILi2ELb1EEEE_.anon"* byval(%"class._ZTSZZ4mainENK3$_0clERN2cl4sycl7handlerEEUlNS1_4itemILi2ELb1EEEE_.anon") align 1 %_arg_1) #0 {
entry:
  %0 = getelementptr inbounds %"class._ZTSN2cl4sycl5rangeILi2EEE.cl::sycl::range", %"class._ZTSN2cl4sycl5rangeILi2EEE.cl::sycl::range"* %_arg_, i64 0, i32 0, i32 0, i64 0
  %.sroa.0.0..sroa_cast9 = addrspacecast i64* %0 to i64 addrspace(4)*
  %.sroa.0.0.copyload10 = load i64, i64 addrspace(4)* %.sroa.0.0..sroa_cast9, align 8
  %1 = load <3 x i64>, <3 x i64> addrspace(4)* addrspacecast (<3 x i64> addrspace(1)* @__spirv_BuiltInGlobalInvocationId to <3 x i64> addrspace(4)*), align 32
  %2 = extractelement <3 x i64> %1, i64 1
  %cmp.i.i = icmp ult i64 %2, 2147483648
  tail call void @llvm.assume(i1 %cmp.i.i)
  %cmp.not.i = icmp ult i64 %2, %.sroa.0.0.copyload10
  br i1 %cmp.not.i, label %if.end.i, label %"_ZZN2cl4sycl7handler27getRangeRoundedKernelLambdaINS0_4itemILi2ELb1EEELi2EZZ4mainENK3$_0clERS1_EUlS4_E_LPv0EEEDaT1_NS0_5rangeIXT0_EEEENKUlS4_E_clES4_.exit"

if.end.i:                                         ; preds = %entry
  tail call spir_func void @__assert_fail(i8 addrspace(4)* getelementptr inbounds ([2 x i8], [2 x i8] addrspace(4)* addrspacecast ([2 x i8] addrspace(1)* @.str to [2 x i8] addrspace(4)*), i64 0, i64 0), i8 addrspace(4)* getelementptr inbounds ([11 x i8], [11 x i8] addrspace(4)* addrspacecast ([11 x i8] addrspace(1)* @.str.1 to [11 x i8] addrspace(4)*), i64 0, i64 0), i32 8, i8 addrspace(4)* getelementptr inbounds ([11 x i8], [11 x i8] addrspace(4)* addrspacecast ([11 x i8] addrspace(1)* @__PRETTY_FUNCTION__._Z3foov to [11 x i8] addrspace(4)*), i64 0, i64 0))
  br label %"_ZZN2cl4sycl7handler27getRangeRoundedKernelLambdaINS0_4itemILi2ELb1EEELi2EZZ4mainENK3$_0clERS1_EUlS4_E_LPv0EEEDaT1_NS0_5rangeIXT0_EEEENKUlS4_E_clES4_.exit"

"_ZZN2cl4sycl7handler27getRangeRoundedKernelLambdaINS0_4itemILi2ELb1EEELi2EZZ4mainENK3$_0clERS1_EUlS4_E_LPv0EEEDaT1_NS0_5rangeIXT0_EEEENKUlS4_E_clES4_.exit": ; preds = %entry, %if.end.i
  ret void
}

; Function Attrs: inaccessiblememonly nofree nosync nounwind willreturn
declare void @llvm.assume(i1 noundef) #1

; Function Attrs: convergent norecurse nounwind mustprogress
define dso_local spir_func void @_Z3foov() {
entry:
  tail call spir_func void @__assert_fail(i8 addrspace(4)* getelementptr inbounds ([2 x i8], [2 x i8] addrspace(4)* addrspacecast ([2 x i8] addrspace(1)* @.str to [2 x i8] addrspace(4)*), i64 0, i64 0), i8 addrspace(4)* getelementptr inbounds ([11 x i8], [11 x i8] addrspace(4)* addrspacecast ([11 x i8] addrspace(1)* @.str.1 to [11 x i8] addrspace(4)*), i64 0, i64 0), i32 8, i8 addrspace(4)* getelementptr inbounds ([11 x i8], [11 x i8] addrspace(4)* addrspacecast ([11 x i8] addrspace(1)* @__PRETTY_FUNCTION__._Z3foov to [11 x i8] addrspace(4)*), i64 0, i64 0))
  ret void
}

; CHECK: _ZTSZZ4mainENK3$_0clERN2cl4sycl7handlerEE9TheKernel
; Function Attrs: convergent norecurse
define weak_odr dso_local spir_kernel void @"_ZTSZZ4mainENK3$_0clERN2cl4sycl7handlerEE9TheKernel"() #0 {
entry:
  tail call spir_func void @__assert_fail(i8 addrspace(4)* getelementptr inbounds ([2 x i8], [2 x i8] addrspace(4)* addrspacecast ([2 x i8] addrspace(1)* @.str to [2 x i8] addrspace(4)*), i64 0, i64 0), i8 addrspace(4)* getelementptr inbounds ([11 x i8], [11 x i8] addrspace(4)* addrspacecast ([11 x i8] addrspace(1)* @.str.1 to [11 x i8] addrspace(4)*), i64 0, i64 0), i32 8, i8 addrspace(4)* getelementptr inbounds ([11 x i8], [11 x i8] addrspace(4)* addrspacecast ([11 x i8] addrspace(1)* @__PRETTY_FUNCTION__._Z3foov to [11 x i8] addrspace(4)*), i64 0, i64 0))
  ret void
}

; CHECK-NOT: _ZTSN2cl4sycl6detail19__pf_kernel_wrapperIZZ4mainENK3$_0clERNS0_7handlerEE10TheKernel2EE
; Function Attrs: norecurse
define weak_odr dso_local spir_kernel void @"_ZTSN2cl4sycl6detail19__pf_kernel_wrapperIZZ4mainENK3$_0clERNS0_7handlerEE10TheKernel2EE"(%"class._ZTSN2cl4sycl5rangeILi2EEE.cl::sycl::range"* byval(%"class._ZTSN2cl4sycl5rangeILi2EEE.cl::sycl::range") align 8 %_arg_, %"class._ZTSZZ4mainENK3$_0clERN2cl4sycl7handlerEEUlNS1_4itemILi2ELb1EEEE_.anon"* byval(%"class._ZTSZZ4mainENK3$_0clERN2cl4sycl7handlerEEUlNS1_4itemILi2ELb1EEEE_.anon") align 1 %_arg_1) #1 {
entry:
  %0 = load <3 x i64>, <3 x i64> addrspace(4)* addrspacecast (<3 x i64> addrspace(1)* @__spirv_BuiltInGlobalInvocationId to <3 x i64> addrspace(4)*), align 32
  %1 = extractelement <3 x i64> %0, i64 1
  %cmp.i.i = icmp ult i64 %1, 2147483648
  tail call void @llvm.assume(i1 %cmp.i.i)
  ret void
}

; Function Attrs: nofree norecurse nosync nounwind readnone willreturn mustprogress
define dso_local spir_func void @_Z3barv() {
entry:
  ret void
}

; CHECK-NOT: _ZTSZZ4mainENK3$_0clERN2cl4sycl7handlerEE10TheKernel2
; Function Attrs: norecurse
define weak_odr dso_local spir_kernel void @"_ZTSZZ4mainENK3$_0clERN2cl4sycl7handlerEE10TheKernel2"() #1 {
entry:
  ret void
}

; CHECK: _ZTSN2cl4sycl6detail19__pf_kernel_wrapperIZZ4mainENK3$_0clERNS0_7handlerEE10TheKernel3EE
; Function Attrs: convergent norecurse
define weak_odr dso_local spir_kernel void @"_ZTSN2cl4sycl6detail19__pf_kernel_wrapperIZZ4mainENK3$_0clERNS0_7handlerEE10TheKernel3EE"(%"class._ZTSN2cl4sycl5rangeILi2EEE.cl::sycl::range"* byval(%"class._ZTSN2cl4sycl5rangeILi2EEE.cl::sycl::range") align 8 %_arg_, %"class._ZTSZZ4mainENK3$_0clERN2cl4sycl7handlerEEUlNS1_4itemILi2ELb1EEEE_.anon"* byval(%"class._ZTSZZ4mainENK3$_0clERN2cl4sycl7handlerEEUlNS1_4itemILi2ELb1EEEE_.anon") align 1 %_arg_1) #0 {
entry:
  %0 = getelementptr inbounds %"class._ZTSN2cl4sycl5rangeILi2EEE.cl::sycl::range", %"class._ZTSN2cl4sycl5rangeILi2EEE.cl::sycl::range"* %_arg_, i64 0, i32 0, i32 0, i64 0
  %.sroa.0.0..sroa_cast9 = addrspacecast i64* %0 to i64 addrspace(4)*
  %.sroa.0.0.copyload10 = load i64, i64 addrspace(4)* %.sroa.0.0..sroa_cast9, align 8
  %1 = load <3 x i64>, <3 x i64> addrspace(4)* addrspacecast (<3 x i64> addrspace(1)* @__spirv_BuiltInGlobalInvocationId to <3 x i64> addrspace(4)*), align 32
  %2 = extractelement <3 x i64> %1, i64 1
  %cmp.i.i = icmp ult i64 %2, 2147483648
  tail call void @llvm.assume(i1 %cmp.i.i)
  %cmp.not.i = icmp ult i64 %2, %.sroa.0.0.copyload10
  br i1 %cmp.not.i, label %if.end.i, label %"_ZZN2cl4sycl7handler27getRangeRoundedKernelLambdaINS0_4itemILi2ELb1EEELi2EZZ4mainENK3$_0clERS1_EUlS4_E1_LPv0EEEDaT1_NS0_5rangeIXT0_EEEENKUlS4_E_clES4_.exit"

if.end.i:                                         ; preds = %entry
  tail call spir_func void @__assert_fail(i8 addrspace(4)* getelementptr inbounds ([2 x i8], [2 x i8] addrspace(4)* addrspacecast ([2 x i8] addrspace(1)* @.str to [2 x i8] addrspace(4)*), i64 0, i64 0), i8 addrspace(4)* getelementptr inbounds ([11 x i8], [11 x i8] addrspace(4)* addrspacecast ([11 x i8] addrspace(1)* @.str.1 to [11 x i8] addrspace(4)*), i64 0, i64 0), i32 8, i8 addrspace(4)* getelementptr inbounds ([11 x i8], [11 x i8] addrspace(4)* addrspacecast ([11 x i8] addrspace(1)* @__PRETTY_FUNCTION__._Z3foov to [11 x i8] addrspace(4)*), i64 0, i64 0))
  br label %"_ZZN2cl4sycl7handler27getRangeRoundedKernelLambdaINS0_4itemILi2ELb1EEELi2EZZ4mainENK3$_0clERS1_EUlS4_E1_LPv0EEEDaT1_NS0_5rangeIXT0_EEEENKUlS4_E_clES4_.exit"

"_ZZN2cl4sycl7handler27getRangeRoundedKernelLambdaINS0_4itemILi2ELb1EEELi2EZZ4mainENK3$_0clERS1_EUlS4_E1_LPv0EEEDaT1_NS0_5rangeIXT0_EEEENKUlS4_E_clES4_.exit": ; preds = %entry, %if.end.i
  ret void
}

; Function Attrs: convergent norecurse nounwind mustprogress
define dso_local spir_func void @_Z3bazv() {
entry:
  tail call spir_func void @__assert_fail(i8 addrspace(4)* getelementptr inbounds ([2 x i8], [2 x i8] addrspace(4)* addrspacecast ([2 x i8] addrspace(1)* @.str to [2 x i8] addrspace(4)*), i64 0, i64 0), i8 addrspace(4)* getelementptr inbounds ([11 x i8], [11 x i8] addrspace(4)* addrspacecast ([11 x i8] addrspace(1)* @.str.1 to [11 x i8] addrspace(4)*), i64 0, i64 0), i32 8, i8 addrspace(4)* getelementptr inbounds ([11 x i8], [11 x i8] addrspace(4)* addrspacecast ([11 x i8] addrspace(1)* @__PRETTY_FUNCTION__._Z3foov to [11 x i8] addrspace(4)*), i64 0, i64 0))
  ret void
}

; CHECK: _ZTSZZ4mainENK3$_0clERN2cl4sycl7handlerEE10TheKernel3
; Function Attrs: convergent norecurse
define weak_odr dso_local spir_kernel void @"_ZTSZZ4mainENK3$_0clERN2cl4sycl7handlerEE10TheKernel3"() #0 {
entry:
  tail call spir_func void @__assert_fail(i8 addrspace(4)* getelementptr inbounds ([2 x i8], [2 x i8] addrspace(4)* addrspacecast ([2 x i8] addrspace(1)* @.str to [2 x i8] addrspace(4)*), i64 0, i64 0), i8 addrspace(4)* getelementptr inbounds ([11 x i8], [11 x i8] addrspace(4)* addrspacecast ([11 x i8] addrspace(1)* @.str.1 to [11 x i8] addrspace(4)*), i64 0, i64 0), i32 8, i8 addrspace(4)* getelementptr inbounds ([11 x i8], [11 x i8] addrspace(4)* addrspacecast ([11 x i8] addrspace(1)* @__PRETTY_FUNCTION__._Z3foov to [11 x i8] addrspace(4)*), i64 0, i64 0))
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
