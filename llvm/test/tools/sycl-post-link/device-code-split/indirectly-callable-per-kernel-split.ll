; RUN: sycl-post-link -split=kernel -S < %s -o %t.table
;
; This test checks that functions marked with "indirectly-callable" LLVM IR
; attribute are outlined into separate device image(s) in accordance with the
; attribute value.
;
; This version of the test is focused on per-kernel device code split
;
; RUN: FileCheck %s --input-file=%t_0.ll --check-prefix CHECK-IR0 \
; RUN:     --implicit-check-not foo --implicit-check-not bar \
; RUN:     --implicit-check-not baz
; RUN: FileCheck %s --input-file=%t_1.ll --check-prefix CHECK-IR1 \
; RUN:     --implicit-check-not kernel --implicit-check-not bar \
; RUN:     --implicit-check-not baz
; RUN: FileCheck %s --input-file=%t_2.ll --check-prefix CHECK-IR2 \
; RUN:     --implicit-check-not kernel --implicit-check-not foo \
; RUN:     --implicit-check-not bar
; RUN: FileCheck %s --input-file=%t_3.ll --check-prefix CHECK-IR3 \
; RUN:     --implicit-check-not kernel --implicit-check-not foo \
; RUN:     --implicit-check-not baz
;
; RUN: sycl-module-split -split=kernel -S < %s -o %t2
; RUN: FileCheck %s --input-file=%t2_0.ll --check-prefix CHECK-IR0 \
; RUN:     --implicit-check-not foo --implicit-check-not bar \
; RUN:     --implicit-check-not baz
; RUN: FileCheck %s --input-file=%t2_1.ll --check-prefix CHECK-IR1 \
; RUN:     --implicit-check-not kernel --implicit-check-not bar \
; RUN:     --implicit-check-not baz
; RUN: FileCheck %s --input-file=%t2_2.ll --check-prefix CHECK-IR2 \
; RUN:     --implicit-check-not kernel --implicit-check-not foo \
; RUN:     --implicit-check-not bar
; RUN: FileCheck %s --input-file=%t2_3.ll --check-prefix CHECK-IR3 \
; RUN:     --implicit-check-not kernel --implicit-check-not foo \
; RUN:     --implicit-check-not baz
;
; CHECK-IR0: define weak_odr dso_local spir_kernel void @kernel
; CHECK-IR1: define spir_func void @foo
; CHECK-IR2: define spir_func void @baz
; CHECK-IR3: define spir_func void @bar

target datalayout = "e-i64:64-v16:16-v24:32-v32:32-v48:64-v96:128-v192:256-v256:256-v512:512-v1024:1024-n8:16:32:64-G1"
target triple = "spir64-unknown-unknown"

define spir_func void @foo() #0 {
entry:
  ret void
}

define spir_func void @bar() #1 {
entry:
  ret void
}

define spir_func void @baz() #1 {
entry:
  ret void
}

define weak_odr dso_local spir_kernel void @kernel() #2 {
entry:
  ret void
}

attributes #0 = { "indirectly-callable"="set-1" "sycl-module-id"="v.cpp" }
attributes #1 = { "indirectly-callable"="set-2" "sycl-module-id"="v.cpp" }
attributes #2 = { "sycl-module-id"="v.cpp" }

