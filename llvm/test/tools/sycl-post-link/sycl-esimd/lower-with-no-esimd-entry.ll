; This test checks to see if ESIMD lowering is performed even without the
; the presence of ESIMD entry points.

; RUN: sycl-post-link -properties -symbols -lower-esimd -split=auto -S < %s -o %t.table
; RUN: FileCheck %s -input-file=%t.table --check-prefixes CHECK-TABLE
; RUN: FileCheck %s -input-file=%t_0.sym --check-prefixes CHECK-SYMS
; RUN: FileCheck %s -input-file=%t_0.ll

; CHECK-TABLE: {{.*}}_0.ll|{{.*}}_0.prop|{{.*}}_0.sym
; CHECK-TABLE-EMPTY:

; CHECK-SYMS: _ZTSZ4mainE3Foo
; CHECK-SYMS-EMPTY:

define weak_odr dso_local spir_kernel void @_ZTSZ4mainE3Foo(ptr addrspace(1) noundef align 4 %_arg_p) #0 {
entry:
  %0 = load i32, ptr addrspace(1) %_arg_p, align 4
  %call1.i.i = tail call x86_regcallcc noundef i32 @_Z33__regcall3____builtin_invoke_simd1(ptr noundef nonnull @helper, ptr noundef nonnull @ESIMD_function, i32 noundef %0) #5
  store i32 %call1.i.i, ptr addrspace(1) %_arg_p, align 4
  ret void
}

define linkonce_odr dso_local x86_regcallcc <16 x i32> @ESIMD_function(<16 x i32> %x) #1 !sycl_explicit_simd !0 !intel_reqd_sub_group_size !1 {
entry:
  ret <16 x i32> zeroinitializer
}

declare dso_local x86_regcallcc noundef i32 @_Z33__regcall3____builtin_invoke_simd1(ptr noundef, ptr noundef, i32 noundef)

; The generated helper should be inlined with the call to @ESIMD_function.
; CHECK: @helper_{{[0-9]+}}(<16 x i32> %simd_args.coerce)
; CHECK-NEXT: entry:
; CHECK-NEXT: ret <16 x i32> zeroinitializer
define linkonce_odr dso_local x86_regcallcc <16 x i32> @helper(ptr noundef nonnull %f, <16 x i32> %simd_args.coerce) #1 {
entry:
  %call = tail call x86_regcallcc <16 x i32> %f(<16 x i32> %simd_args.coerce)
  ret <16 x i32> %call
}

attributes #0 = { "sycl-module-id"="test.cpp" }
attributes #1 = { "referenced-indirectly" }

!0 = !{}
!1 = !{i32 1}