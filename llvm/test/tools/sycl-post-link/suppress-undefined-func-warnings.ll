; Verify that sycl-post-link's "Undefined function ... found in ..." warning
; can be silenced via -suppress-undefined-func-warnings. This is the raw
; sycl-post-link flag that clang forwards when the user passes
; -Wno-sycl-undefined-func-in-image (see clang/lib/Driver/ToolChains/Clang.cpp).

; Default: warning fires for an undefined, referenced, non-builtin function.
; RUN: sycl-post-link -split=auto -symbols -S < %s -o %t.default.table 2>&1 \
; RUN:   | FileCheck %s --check-prefix=CHECK-WARN

; With the flag: warning is suppressed.
; RUN: sycl-post-link -split=auto -symbols -suppress-undefined-func-warnings \
; RUN:   -S < %s -o %t.suppressed.table 2>&1 \
; RUN:   | FileCheck %s --allow-empty --check-prefix=CHECK-SUPPRESSED

; CHECK-WARN: warning: Undefined function undefined_user_func found in
; CHECK-SUPPRESSED-NOT: warning: Undefined function undefined_user_func found in

target datalayout = "e-i64:64-v16:16-v24:32-v32:32-v48:64-v96:128-v192:256-v256:256-v512:512-v1024:1024"
target triple = "spir64-unknown-linux"

declare spir_func void @undefined_user_func()

define weak_odr dso_local spir_kernel void @kernel() #0 {
  call spir_func void @undefined_user_func()
  ret void
}

attributes #0 = { "sycl-module-id"="a.cpp" }
