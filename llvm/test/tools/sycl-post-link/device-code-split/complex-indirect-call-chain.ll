; The idea of the test is to ensure that sycl-post-link can trace through more
; complex call stacks involving several nested indirect calls

; RUN: sycl-post-link -split=auto -S < %s -o %t.table
; RUN: FileCheck %s -input-file=%t_0.ll --check-prefix CHECK0 \
; RUN:     --implicit-check-not @foo --implicit-check-not @kernel_A \
; RUN:     --implicit-check-not @kernel_B --implicit-check-not @baz
; RUN: FileCheck %s -input-file=%t_1.ll --check-prefix CHECK1 \
; RUN:     --implicit-check-not @kernel_A --implicit-check-not @kernel_C
; RUN: FileCheck %s -input-file=%t_2.ll --check-prefix CHECK2 \
; RUN:     --implicit-check-not @foo --implicit-check-not @bar \
; RUN:     --implicit-check-not @BAZ --implicit-check-not @kernel_B \
; RUN:     --implicit-check-not @kernel_C
;
; RUN: sycl-post-link -split=source -S < %s -o %t.table
; RUN: FileCheck %s -input-file=%t_0.ll --check-prefix CHECK0 \
; RUN:     --implicit-check-not @foo --implicit-check-not @kernel_A \
; RUN:     --implicit-check-not @kernel_B --implicit-check-not @baz
; RUN: FileCheck %s -input-file=%t_1.ll --check-prefix CHECK1 \
; RUN:     --implicit-check-not @kernel_A --implicit-check-not @kernel_C
; RUN: FileCheck %s -input-file=%t_2.ll --check-prefix CHECK2 \
; RUN:     --implicit-check-not @foo --implicit-check-not @bar \
; RUN:     --implicit-check-not @BAZ --implicit-check-not @kernel_B \
; RUN:     --implicit-check-not @kernel_C
;
; RUN: sycl-post-link -split=kernel -S < %s -o %t.table
; RUN: FileCheck %s -input-file=%t_0.ll --check-prefix CHECK0 \
; RUN:     --implicit-check-not @foo --implicit-check-not @kernel_A \
; RUN:     --implicit-check-not @kernel_B --implicit-check-not @baz
; RUN: FileCheck %s -input-file=%t_1.ll --check-prefix CHECK1 \
; RUN:     --implicit-check-not @kernel_A --implicit-check-not @kernel_C
; RUN: FileCheck %s -input-file=%t_2.ll --check-prefix CHECK2 \
; RUN:     --implicit-check-not @foo --implicit-check-not @bar \
; RUN:     --implicit-check-not @BAZ --implicit-check-not @kernel_B \
; RUN:     --implicit-check-not @kernel_C

; CHECK0-DAG: define spir_kernel void @kernel_C
; CHECK0-DAG: define spir_func i32 @bar
; CHECK0-DAG: define spir_func void @BAZ

; CHECK1-DAG: define spir_kernel void @kernel_B
; CHECK1-DAG: define spir_func i32 @foo
; CHECK1-DAG: define spir_func i32 @bar
; CHECK1-DAG: define spir_func void @BAZ

; CHECK2-DAG: define spir_kernel void @kernel_A
; CHECK2-DAG: define spir_func void @baz

target datalayout = "e-i64:64-v16:16-v24:32-v32:32-v48:64-v96:128-v192:256-v256:256-v512:512-v1024:1024-n8:16:32:64"
target triple = "spir64-unknown-unknown"

define spir_func i32 @foo(i32 (i32, void ()*)* %ptr1, void ()* %ptr2) {
  %1 = call spir_func i32 %ptr1(i32 42, void ()* %ptr2)
  ret i32 %1
}

define spir_func i32 @bar(i32 %arg, void ()* %ptr) #3 {
  call spir_func void %ptr()
  ret i32 %arg
}

define spir_func void @baz() {
  ret void
}

define spir_func void @BAZ() #3 {
  ret void
}

define spir_kernel void @kernel_A() #0 {
  call spir_func void @baz()
  ret void
}

define spir_kernel void @kernel_B() #1 {
  call spir_func i32 @foo(i32 (i32, void ()*)* null, void ()* null)
  ret void
}

define spir_kernel void @kernel_C() #2 {
  call spir_func i32 @bar(i32 42, void ()* null)
  ret void
}

attributes #0 = { "sycl-module-id"="TU1.cpp" }
attributes #1 = { "sycl-module-id"="TU2.cpp" }
attributes #2 = { "sycl-module-id"="TU3.cpp" }
attributes #3 = { "referenced-indirectly" }
