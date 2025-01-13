; RUN: sycl-post-link -properties --emit-only-kernels-as-entry-points -symbols -split=auto -S < %s -o %t.table
; RUN: FileCheck %s -input-file=%t.table --check-prefixes CHECK-TABLE
; RUN: FileCheck %s -input-file=%t_0.sym --check-prefixes CHECK-M0-SYMS
; RUN: FileCheck %s -input-file=%t_1.sym --check-prefixes CHECK-M1-SYMS
; RUN: FileCheck %s -input-file=%t_1.ll --implicit-check-not double

; Two module should be generated, one contains double kernel, other contains float kernel
; CHECK-TABLE: {{.*}}_0.ll|{{.*}}_0.prop|{{.*}}_0.sym
; CHECK-TABLE: {{.*}}_1.ll|{{.*}}_1.prop|{{.*}}_1.sym
; CHECK-TABLE-EMPTY:

; CHECK-M0-SYMS: double_kernel
; CHECK-M0-SYMS-EMPTY:

; CHECK-M1-SYMS: float_kernel
; CHECK-M1-SYMS-EMPTY:

target datalayout = "e-i64:64-v16:16-v24:32-v32:32-v48:64-v96:128-v192:256-v256:256-v512:512-v1024:1024-n8:16:32:64"
target triple = "spir64-unknown-unknown"

; ================ float kernel ================

declare dso_local x86_regcallcc noundef float @_Z33__regcall3____builtin_invoke_simdfloat(ptr noundef, ptr noundef, float noundef, float noundef)

define weak_odr dso_local spir_kernel void @float_kernel() #0 {
entry:
  %res = tail call x86_regcallcc noundef float @_Z33__regcall3____builtin_invoke_simdfloat(ptr noundef nonnull @helper_float, ptr noundef nonnull @simd_func_float, float noundef 1.0, float noundef 2.0)
  ret void
}

define linkonce_odr dso_local x86_regcallcc <2 x float> @helper_float(ptr noundef nonnull %f, <2 x float> %simd_args.coerce, float noundef %simd_args3) #0 {
entry:
  %call = tail call x86_regcallcc <2 x float> %f(<2 x float> %simd_args.coerce, float noundef %simd_args3)
  ret <2 x float> %call
}

define linkonce_odr dso_local x86_regcallcc <2 x float> @simd_func_float(<2 x float> %x.coerce, float noundef %n) #0 !sycl_explicit_simd !0 !intel_reqd_sub_group_size !1 {
entry:
  ret <2 x float> zeroinitializer
}

; ================ double kernel ================

declare dso_local x86_regcallcc noundef double @_Z33__regcall3____builtin_invoke_simddouble(ptr noundef, ptr noundef, double noundef, double noundef)

define weak_odr dso_local spir_kernel void @double_kernel() #0 !sycl_used_aspects !2 {
entry:
  %res = tail call x86_regcallcc noundef double @_Z33__regcall3____builtin_invoke_simddouble(ptr noundef nonnull @helper_double, ptr noundef nonnull @simd_func_double, double noundef 1.0, double noundef 2.0)
  ret void
}

define linkonce_odr dso_local x86_regcallcc <2 x double> @helper_double(ptr noundef nonnull %f, <2 x double> %simd_args.coerce, double noundef %simd_args3) #0 !sycl_used_aspects !2 {
entry:
  %call = tail call x86_regcallcc <2 x double> %f(<2 x double> %simd_args.coerce, double noundef %simd_args3)
  ret <2 x double> %call
}

define linkonce_odr dso_local x86_regcallcc <2 x double> @simd_func_double(<2 x double> %x.coerce, double noundef %n) #0 !sycl_explicit_simd !0 !intel_reqd_sub_group_size !1 !sycl_used_aspects !2 {
entry:
  ret <2 x double> zeroinitializer
}

attributes #0 = { "sycl-module-id"="TU1.cpp" }

!0 = !{}
!1 = !{i32 1}
!2 = !{i32 6}
