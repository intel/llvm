; Verify that -undefined-symbols-are-errors reports undefined functions and
; global variables as errors, replacing sycl-post-link's "Undefined function ...
; found in ..." warning. Intrinsics, builtins and SPIR-V built-in variables are
; not reported.

; RUN: not sycl-post-link -split=auto -symbols -undefined-symbols-are-errors \
; RUN:   -S < %s -o %t.auto.table 2>&1 | FileCheck %s --check-prefix=CHECK-ERROR
; RUN: not sycl-post-link -split=none -symbols -undefined-symbols-are-errors \
; RUN:   -S < %s -o %t.none.table 2>&1 | FileCheck %s --check-prefix=CHECK-ERROR

; With dynamic linking of device images, undefined symbols are allowed.
; RUN: sycl-post-link -split=auto -symbols -undefined-symbols-are-errors \
; RUN:   -allow-device-image-dependencies -S < %s -o %t.dependencies.table 2>&1 \
; RUN:   | FileCheck %s --allow-empty --check-prefix=CHECK-NONE

; CHECK-ERROR-NOT: warning: Undefined function
; CHECK-ERROR: error: undefined symbol(s) undefined_user_func, undefined_user_global found in
; CHECK-NONE-NOT: undefined_user

target datalayout = "e-i64:64-v16:16-v24:32-v32:32-v48:64-v96:128-v192:256-v256:256-v512:512-v1024:1024"
target triple = "spir64-unknown-linux"

declare spir_func void @undefined_user_func()
declare spir_func i64 @_Z28__spirv_GlobalInvocationId_xv()
declare void @llvm.assume(i1)

@undefined_user_global = external addrspace(1) global i32
@__spirv_BuiltInGlobalInvocationId = external addrspace(1) constant <3 x i64>

define weak_odr dso_local spir_kernel void @kernel() #0 {
  call spir_func void @undefined_user_func()
  %id = call spir_func i64 @_Z28__spirv_GlobalInvocationId_xv()
  call void @llvm.assume(i1 true)
  %val = load i32, ptr addrspace(1) @undefined_user_global
  %gid = load <3 x i64>, ptr addrspace(1) @__spirv_BuiltInGlobalInvocationId
  ret void
}

attributes #0 = { "sycl-module-id"="a.cpp" }
