; RUN: opt -passes=LowerESIMD -S < %s | FileCheck %s

; This test checks that LowerESIMD pass inlines all non-kernel functions.
; If the function has noinline attribute -- honor that,
; unless... it has a call of slm_init() and is called from one spir_kernel.

define spir_kernel void @EsimdKernel1() !sycl_explicit_simd !0 {
; CHECK: @EsimdKernel1
; CHECK-NEXT:    call void @noinline_func()
; CHECK-NEXT:    ret void
  call void @foo()
  call void @bar()
  ret void
}

define spir_kernel void @EsimdKernel2() !sycl_explicit_simd !0 {
; CHECK: @EsimdKernel2
; CHECK-NEXT:    call void @noinline_func()
; CHECK-NEXT:    ret void
  call void @foobar()
  call void @noinline_func()
  ret void
}

define spir_kernel void @EsimdKernel3() !sycl_explicit_simd !0 {
; CHECK: @EsimdKernel3
; CHECK-NEXT:    [[I32VAR:%[a-zA-Z0-9.]+]] = load i32, ptr @c, align 4
; CHECK-NEXT:    call void @llvm.genx.slm.init(i32 [[I32VAR]]) 
; CHECK-NEXT:    call void @noinline_func()
; CHECK-NEXT:    ret void
  call void @slm_init_caller_ignore_noinline()
  call void @noinline_func()
  ret void
}

; First kernel calling spir_func slm_init_caller_noinline_with_2_callers
define spir_kernel void @EsimdKernelA() !sycl_explicit_simd !0 {
; CHECK:      @EsimdKernelA
; CHECK-NEXT: call void @slm_init_caller_noinline_with_2_callers()
; CHECK-NEXT: ret void
  call void @slm_init_caller_noinline_with_2_callers()
  ret void
}

; Second kernel calling spir_func slm_init_caller_noinline_with_2_callers
define spir_kernel void @EsimdKernelB() !sycl_explicit_simd !0 {
; CHECK:      @EsimdKernelB
; CHECK-NEXT: call void @slm_init_caller_noinline_with_2_callers()
; CHECK-NEXT: ret void
  call void @slm_init_caller_noinline_with_2_callers()
  ret void
}

declare dso_local spir_func void @_Z16__esimd_slm_initj(i32 noundef) local_unnamed_addr
; CHECK: declare dso_local spir_func void @_Z16__esimd_slm_initj(i32 noundef) local_unnamed_addr #[[ATTR_INL:[0-9]+]]

@c = dso_local global i32 0, align 4

define spir_func void @slm_init_caller_ignore_noinline() #0 {
; CHECK:      define spir_func void @slm_init_caller_ignore_noinline() #[[ATTR_INL]]
; CHECK-NEXT: [[I32VAR2:%[a-zA-Z0-9.]+]] = load i32, ptr @c, align 4
; CHECK-NEXT: call void @llvm.genx.slm.init(i32 [[I32VAR2]])
; CHECK-NEXT: ret void
  %v = load i32, ptr @c, align 4
  call spir_func void @_Z16__esimd_slm_initj(i32 %v)
  ret void
}

define spir_func void @slm_init_caller_noinline_with_2_callers() #0 {
; CHECK: define spir_func void @slm_init_caller_noinline_with_2_callers() #[[ATTR_NOINL:[0-9]+]]
; CHECK: ret void
  %v = load i32, ptr @c, align 4
  call spir_func void @_Z16__esimd_slm_initj(i32 %v)
  ret void
}

define spir_func void @foo() {
; CHECK:      define spir_func void @foo() #[[ATTR_INL]]
; CHECK-NEXT: ret void
  ret void
}

define spir_func void @bar() {
; CHECK:      define spir_func void @bar() #[[ATTR_INL]]
; CHECK-NEXT: call void @noinline_func()
; CHECK-NEXT: ret void
  call void @foobar()
  call void @noinline_func()
  ret void
}

define spir_func void @foobar() {
; CHECK:      define spir_func void @foobar() #[[ATTR_INL]]
; CHECK-NEXT: ret void
  ret void
}

define spir_func void @noinline_func() #0 {
; CHECK: @noinline_func() #[[ATTR_NOINL]] {
; CHECK-NEXT: ret void
  ret void
}

!0 = !{}

attributes #0 = { noinline }
; CHECK-DAG: attributes #[[ATTR_INL]] = { alwaysinline }
; CHECK-DAG: attributes #[[ATTR_NOINL]] = { noinline }
