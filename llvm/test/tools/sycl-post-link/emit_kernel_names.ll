; This test checks that the post-link tool generates list of kernel names.
;
; Global scope
; RUN: sycl-post-link -properties -symbols -emit-kernel-names -S < %s -o %t.global.files.table
; RUN: FileCheck %s -input-file=%t.global.files_0.prop --implicit-check-not="SpirFunc" --check-prefix=CHECK-GLOBAL-PROP
;
; Per-module split
; RUN: sycl-post-link -properties -symbols -split=source -emit-kernel-names -S < %s -o %t.per_module.files.table
; RUN: FileCheck %s -input-file=%t.per_module.files_0.prop -implicit-check-not="SpirFunc" --check-prefix=CHECK-PERMODULE-0-PROP
; RUN: FileCheck %s -input-file=%t.per_module.files_1.prop -implicit-check-not="SpirFunc" --check-prefix=CHECK-PERMODULE-1-PROP
; RUN: FileCheck %s -input-file=%t.per_module.files_2.prop -implicit-check-not="SpirFunc" --check-prefix=CHECK-KERNELLESS-PROP
;
; Per-kernel split
; RUN: sycl-post-link -properties -symbols -split=kernel -emit-kernel-names -S < %s -o %t.per_kernel.files.table
; RUN: FileCheck %s -input-file=%t.per_kernel.files_0.prop --implicit-check-not="SpirFunc" --check-prefix=CHECK-PERKERNEL-0-PROP
; RUN: FileCheck %s -input-file=%t.per_kernel.files_1.prop --implicit-check-not="SpirFunc" --check-prefix=CHECK-PERKERNEL-1-PROP
; RUN: FileCheck %s -input-file=%t.per_kernel.files_2.prop --implicit-check-not="SpirFunc" --check-prefix=CHECK-PERKERNEL-2-PROP
; RUN: FileCheck %s -input-file=%t.per_kernel.files_3.prop --implicit-check-not="SpirFunc" --check-prefix=CHECK-KERNELLESS-PROP
; RUN: FileCheck %s -input-file=%t.per_kernel.files_4.prop --implicit-check-not="SpirFunc" --check-prefix=CHECK-KERNELLESS-PROP
; RUN: FileCheck %s -input-file=%t.per_kernel.files_5.prop --implicit-check-not="SpirFunc" --check-prefix=CHECK-KERNELLESS-PROP

target triple = "spir64-unknown-unknown"

define dso_local spir_kernel void @SpirKernel1(float %arg1) #2 {
entry:
  ret void
}

define dso_local spir_kernel void @SpirKernel2(float %arg1) #1 {
entry:
  ret void
}

define dso_local spir_kernel void @SpirKernel3(float %arg1) #2 {
entry:
  ret void
}

define dso_local spir_func void @SpirFunc1(float %arg1) #0 {
entry:
  ret void
}

define dso_local spir_func void @SpirFunc2(i32 %arg1, i32 %arg2) #1 {
entry:
  ret void
}

define dso_local spir_func void @SpirFunc3(float %arg1) #0 {
entry:
  ret void
}

define dso_local spir_func void @SpirFunc4(float %arg1) {
entry:
  ret void
}

attributes #0 = { "sycl-module-id"="a.cpp" }
attributes #1 = { "sycl-module-id"="b.cpp" }
attributes #2 = { "sycl-module-id"="c.cpp" }

; Global scope
; CHECK-GLOBAL-PROP: [SYCL/kernel names]
; CHECK-GLOBAL-PROP-NEXT: SpirKernel1
; CHECK-GLOBAL-PROP-NEXT: SpirKernel2
; CHECK-GLOBAL-PROP-NEXT: SpirKernel3

; Per-module split
; CHECK-PERMODULE-0-PROP: [SYCL/kernel names]
; CHECK-PERMODULE-0-PROP-NEXT: SpirKernel1
; CHECK-PERMODULE-0-PROP-NEXT: SpirKernel3
; CHECK-PERMODULE-0-PROP-NOT: SpirKernel2

; CHECK-PERMODULE-1-PROP: [SYCL/kernel names]
; CHECK-PERMODULE-1-PROP-NEXT: SpirKernel2
; CHECK-PERMODULE-1-PROP-NOT: SpirKernel1
; CHECK-PERMODULE-1-PROP-NOT: SpirKernel3

; Per-kernel split
; CHECK-PERKERNEL-0-PROP: [SYCL/kernel names]
; CHECK-PERKERNEL-0-PROP-NEXT: SpirKernel3
; CHECK-PERKERNEL-0-PROP-NOT: SpirKernel1
; CHECK-PERKERNEL-0-PROP-NOT: SpirKernel2

; CHECK-PERKERNEL-1-PROP: [SYCL/kernel names]
; CHECK-PERKERNEL-1-PROP-NEXT: SpirKernel2
; CHECK-PERKERNEL-1-PROP-NOT: SpirKernel1
; CHECK-PERKERNEL-1-PROP-NOT: SpirKernel3

; CHECK-PERKERNEL-2-PROP: [SYCL/kernel names]
; CHECK-PERKERNEL-2-PROP-NEXT: SpirKernel1
; CHECK-PERKERNEL-2-PROP-NOT: SpirKernel2
; CHECK-PERKERNEL-2-PROP-NOT: SpirKernel3

; Kernel-less generated modules should have no kernel names
; CHECK-KERNELLESS-PROP-NOT: [SYCL/kernel names]
; CHECK-KERNELLESS-PROP-NOT: SpirKernel1
; CHECK-KERNELLESS-PROP-NOT: SpirKernel2
; CHECK-KERNELLESS-PROP-NOT: SpirKernel3
