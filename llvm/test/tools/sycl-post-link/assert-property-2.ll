; This test checks that the post-link tool properly generates "assert used"
; property - it should include only kernels that call assertions in their call
; graph.

; RUN: sycl-post-link -split=auto -symbols -S %s -o %t.table
; RUN: FileCheck %s -input-file=%t_0.prop -check-prefix=PRESENCE-CHECK
; RUN: FileCheck %s -input-file=%t_0.prop -check-prefix=ABSENCE-CHECK

; SYCL source:
; void assert_func() {
;   assert(0);
; }
;
; void A_excl() {}
; void B_incl() { assert_func(); }
;
; void A_incl() { assert_func(); }
; void B_excl() {}
;
; void C_excl() {}
; void D_incl() { assert_func(); }
; void common() {
;   C_excl();
;   D_incl();
; }
;
; void C_incl() { assert_func(); }
; void D_excl() {}
; void common2() {
;   C_incl();
;   D_excl();
; }
;
; void E_excl() {}
; void F_incl() { assert_func(); }
;
; void I_incl() { assert_func(); }
; void common3() { I_incl();}
; void G() { common3(); }
; void H() { common3(); }
;
; void no_assert_func() {
;   return;
; }
; void common4() {
;   assert_func();
;   no_assert_func();
; }
; void J() {
;   common4();
; }
;
; int main() {
;   queue Q;
;   Q.submit([&] (handler& CGH) {
;     CGH.parallel_for<class Kernel9>(range<1>{1}, [=](id<1> i) {
;       J();
;     });
;     CGH.parallel_for<class Kernel10>(range<1>{1}, [=](id<1> i) {
;       common4();
;     });
;     CGH.parallel_for<class Kernel>(range<1>{1}, [=](id<1> i) {
;       A_excl();
;       B_incl();
;     });
;     CGH.parallel_for<class Kernel2>(range<1>{1}, [=](id<1> i) {
;       A_incl();
;       B_excl();
;     });
;
;     CGH.parallel_for<class Kernel3>(range<1>{1}, [=](id<1> i) {
;       common();
;     });
;     CGH.parallel_for<class Kernel4>(range<1>{1}, [=](id<1> i) {
;       common2();
;     });
;
;     CGH.parallel_for<class Kernel5>(range<1>{1}, [=](id<1> i) {
;       B_incl();
;       A_excl();
;     });
;
;     CGH.parallel_for<class Kernel6>(range<1>{1}, [=](id<1> i) {
;       E_excl();
;       E_excl();
;     });
;     CGH.parallel_for<class Kernel7>(range<1>{1}, [=](id<1> i) {
;       F_incl();
;       F_incl();
;     });
;
;     CGH.parallel_for<class Kernel8>(range<1>{1}, [=](id<1> i) {
;       G();
;       H();
;     });
;   });
;   Q.wait();
;   return 0;
; }

target datalayout = "e-i64:64-v16:16-v24:32-v32:32-v48:64-v96:128-v192:256-v256:256-v512:512-v1024:1024-n8:16:32:64"
target triple = "spir64_x86_64-unknown-unknown"

@.str = private unnamed_addr addrspace(1) constant [2 x i8] c"0\00", align 1
@.str.1 = private unnamed_addr addrspace(1) constant [16 x i8] c"assert_test.cpp\00", align 1
@__PRETTY_FUNCTION__._Z11assert_funcv = private unnamed_addr addrspace(1) constant [19 x i8] c"void assert_func()\00", align 1
@_ZL10assert_fmt = internal addrspace(2) constant [85 x i8] c"%s:%d: %s: global id: [%lu,%lu,%lu], local id: [%lu,%lu,%lu] Assertion `%s` failed.\0A\00", align 1

; PRESENCE-CHECK: [SYCL/assert used]

; Function Attrs: convergent noinline norecurse optnone mustprogress
define dso_local spir_func void @_Z1Jv() #3 {
entry:
  call spir_func void @_Z7common4v()
  ret void
}

; Function Attrs: convergent noinline norecurse optnone mustprogress
define dso_local spir_func void @_Z7common4v() #3 {
entry:
  call spir_func void @_Z11assert_funcv()
  call spir_func void @_Z14no_assert_funcv()
  ret void
}

; PRESENCE-CHECK-DAG: _ZTSZZ4mainENKUlRN2cl4sycl7handlerEE_clES2_E7Kernel9
; Function Attrs: convergent noinline norecurse mustprogress
define weak_odr dso_local spir_kernel void @_ZTSZZ4mainENKUlRN2cl4sycl7handlerEE_clES2_E7Kernel9() #0 {
entry:
  call spir_func void @_Z1Jv()
  ret void
}

; PRESENCE-CHECK-DAG: _ZTSZZ4mainENKUlRN2cl4sycl7handlerEE_clES2_E8Kernel10
; Function Attrs: convergent noinline norecurse optnone mustprogress
define weak_odr dso_local spir_kernel void @_ZTSZZ4mainENKUlRN2cl4sycl7handlerEE_clES2_E8Kernel10() #0 {
entry:
  call spir_func void @_Z7common4v()
  ret void
}

; Function Attrs: convergent noinline norecurse nounwind optnone mustprogress
define dso_local spir_func void @_Z14no_assert_funcv() #2 {
entry:
  ret void
}

; Function Attrs: convergent norecurse nounwind mustprogress
define dso_local spir_func void @_Z6B_inclv() local_unnamed_addr {
entry:
  call spir_func void @_Z11assert_funcv()
  ret void
}

; Function Attrs: convergent norecurse nounwind mustprogress
define dso_local spir_func void @_Z11assert_funcv() local_unnamed_addr {
entry:
  call spir_func void @__assert_fail(i8 addrspace(4)* getelementptr inbounds ([2 x i8], [2 x i8] addrspace(4)* addrspacecast ([2 x i8] addrspace(1)* @.str to [2 x i8] addrspace(4)*), i64 0, i64 0), i8 addrspace(4)* getelementptr inbounds ([16 x i8], [16 x i8] addrspace(4)* addrspacecast ([16 x i8] addrspace(1)* @.str.1 to [16 x i8] addrspace(4)*), i64 0, i64 0), i32 7, i8 addrspace(4)* getelementptr inbounds ([19 x i8], [19 x i8] addrspace(4)* addrspacecast ([19 x i8] addrspace(1)* @__PRETTY_FUNCTION__._Z11assert_funcv to [19 x i8] addrspace(4)*), i64 0, i64 0))
  ret void
}

; Function Attrs: nofree norecurse nosync nounwind readnone willreturn mustprogress
define dso_local spir_func void @_Z6A_exclv() local_unnamed_addr {
entry:
  ret void
}

; PRESENCE-CHECK-DAG: _ZTSZZ4mainENK3$_0clERN2cl4sycl7handlerEE6Kernel
; Function Attrs: convergent norecurse mustprogress
define weak_odr dso_local spir_kernel void @"_ZTSZZ4mainENK3$_0clERN2cl4sycl7handlerEE6Kernel"() local_unnamed_addr #0 {
entry:
  call spir_func void @_Z6A_exclv()
  call spir_func void @_Z6B_inclv()
  ret void
}

; Function Attrs: convergent norecurse nounwind mustprogress
define dso_local spir_func void @_Z6A_inclv() local_unnamed_addr {
entry:
  call spir_func void @_Z11assert_funcv()
  ret void
}

; Function Attrs: nofree norecurse nosync nounwind readnone willreturn mustprogress
define dso_local spir_func void @_Z6B_exclv() local_unnamed_addr {
entry:
  ret void
}

; PRESENCE-CHECK-DAG: _ZTSZZ4mainENK3$_0clERN2cl4sycl7handlerEE7Kernel2
; Function Attrs: convergent norecurse mustprogress
define weak_odr dso_local spir_kernel void @"_ZTSZZ4mainENK3$_0clERN2cl4sycl7handlerEE7Kernel2"() local_unnamed_addr #0 {
entry:
  call spir_func void @_Z6A_inclv()
  call spir_func void @_Z6B_exclv()
  ret void
}

; Function Attrs: convergent norecurse nounwind mustprogress
define dso_local spir_func void @_Z6commonv() local_unnamed_addr {
entry:
  call spir_func void @_Z6C_exclv()
  call spir_func void @_Z6D_inclv()
  ret void
}

; Function Attrs: convergent norecurse nounwind mustprogress
define dso_local spir_func void @_Z6D_inclv() local_unnamed_addr {
entry:
  call spir_func void @_Z11assert_funcv()
  ret void
}

; Function Attrs: nofree norecurse nosync nounwind readnone willreturn mustprogress
define dso_local spir_func void @_Z6C_exclv() local_unnamed_addr {
entry:
  ret void
}

; PRESENCE-CHECK-DAG: _ZTSZZ4mainENK3$_0clERN2cl4sycl7handlerEE7Kernel3
; Function Attrs: convergent norecurse mustprogress
define weak_odr dso_local spir_kernel void @"_ZTSZZ4mainENK3$_0clERN2cl4sycl7handlerEE7Kernel3"() local_unnamed_addr #0 {
entry:
  call spir_func void @_Z6commonv()
  ret void
}

; Function Attrs: convergent norecurse nounwind mustprogress
define dso_local spir_func void @_Z7common2v() local_unnamed_addr {
entry:
  call spir_func void @_Z6C_inclv()
  ret void
}

; Function Attrs: convergent norecurse nounwind mustprogress
define dso_local spir_func void @_Z6C_inclv() local_unnamed_addr {
entry:
  call spir_func void @_Z11assert_funcv()
  ret void
}

; Function Attrs: nofree norecurse nosync nounwind readnone willreturn mustprogress
define dso_local spir_func void @_Z6D_exclv() local_unnamed_addr {
entry:
  ret void
}

; PRESENCE-CHECK-DAG: _ZTSZZ4mainENK3$_0clERN2cl4sycl7handlerEE7Kernel4
; Function Attrs: convergent norecurse mustprogress
define weak_odr dso_local spir_kernel void @"_ZTSZZ4mainENK3$_0clERN2cl4sycl7handlerEE7Kernel4"() local_unnamed_addr #0 {
entry:
  call spir_func void @_Z7common2v()
  ret void
}

; PRESENCE-CHECK-DAG: _ZTSZZ4mainENK3$_0clERN2cl4sycl7handlerEE7Kernel5
; Function Attrs: convergent norecurse mustprogress
define weak_odr dso_local spir_kernel void @"_ZTSZZ4mainENK3$_0clERN2cl4sycl7handlerEE7Kernel5"() local_unnamed_addr #0 {
entry:
  call spir_func void @_Z6B_inclv()
  call spir_func void @_Z6A_exclv()
  ret void
}

; Function Attrs: nofree norecurse nosync nounwind readnone willreturn mustprogress
define dso_local spir_func void @_Z6E_exclv() local_unnamed_addr {
entry:
  ret void
}

; Function Attrs: convergent norecurse nounwind mustprogress
define dso_local spir_func void @_Z6F_inclv() local_unnamed_addr {
entry:
  call spir_func void @_Z11assert_funcv()
  ret void
}

; PRESENCE-CHECK-DAG: _ZTSZZ4mainENK3$_0clERN2cl4sycl7handlerEE7Kernel7
; Function Attrs: convergent norecurse mustprogress
define weak_odr dso_local spir_kernel void @"_ZTSZZ4mainENK3$_0clERN2cl4sycl7handlerEE7Kernel7"() local_unnamed_addr #0 {
entry:
  call spir_func void @_Z6F_inclv()
  call spir_func void @_Z6F_inclv()
  ret void
}

; Function Attrs: convergent inlinehint norecurse nounwind mustprogress
define internal spir_func void @"_ZZZ4mainENK3$_0clERN2cl4sycl7handlerEENKUlNS1_2idILi1EEEE6_clES5_"() unnamed_addr align 2 {
entry:
  call spir_func void @_Z1Gv()
  call spir_func void @_Z1Hv()
  ret void
}

; Function Attrs: convergent norecurse nounwind mustprogress
define dso_local spir_func void @_Z1Gv() local_unnamed_addr {
entry:
  call spir_func void @_Z7common3v()
  ret void
}

; Function Attrs: convergent norecurse nounwind mustprogress
define dso_local spir_func void @_Z1Hv() local_unnamed_addr {
entry:
  call spir_func void @_Z7common3v()
  ret void
}

; Function Attrs: convergent norecurse nounwind mustprogress
define dso_local spir_func void @_Z7common3v() local_unnamed_addr {
entry:
  call spir_func void @_Z6I_inclv()
  ret void
}

; Function Attrs: convergent norecurse nounwind mustprogress
define dso_local spir_func void @_Z6I_inclv() local_unnamed_addr {
entry:
  call spir_func void @_Z11assert_funcv()
  ret void
}

; PRESENCE-CHECK-DAG: _ZTSZZ4mainENK3$_0clERN2cl4sycl7handlerEE7Kernel8
; Function Attrs: convergent norecurse mustprogress
define weak_odr dso_local spir_kernel void @"_ZTSZZ4mainENK3$_0clERN2cl4sycl7handlerEE7Kernel8"() local_unnamed_addr #0 {
  call spir_func void @_Z1Gv()
  call spir_func void @_Z1Hv()
  ret void
}

; ABSENCE-CHECK-NOT: _ZTSZZ4mainENK3$_0clERN2cl4sycl7handlerEE7Kernel6
; Function Attrs: convergent norecurse mustprogress
define weak_odr dso_local spir_kernel void @"_ZTSZZ4mainENK3$_0clERN2cl4sycl7handlerEE7Kernel6"() local_unnamed_addr #0 {
entry:
  call spir_func void @_Z6E_exclv()
  call spir_func void @_Z6E_exclv()
  ret void
}

; Function Attrs: convergent norecurse mustprogress
define weak dso_local spir_func void @__assert_fail(i8 addrspace(4)* %expr, i8 addrspace(4)* %file, i32 %line, i8 addrspace(4)* %func) local_unnamed_addr {
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
declare dso_local spir_func i32 @_Z18__spirv_ocl_printfPU3AS2Kcz(i8 addrspace(2)*, ...) local_unnamed_addr

attributes #0 = { convergent norecurse mustprogress "frame-pointer"="all" "min-legal-vector-width"="0" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "sycl-module-id"="assert_test.cpp" "uniform-work-group-size"="true" }

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
