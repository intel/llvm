; This test checks that the post-link tool generates list of exported symbols.
;
; Global scope
; RUN: sycl-post-link -properties -symbols -emit-exported-symbols -S < %s -o %t.global.files.table
; RUN: FileCheck %s -input-file=%t.global.files_0.prop --implicit-check-not="NotExported" --check-prefix=CHECK-GLOBAL-PROP
;
; Per-module split
; RUN: sycl-post-link -properties -symbols -split=source -emit-exported-symbols -S < %s -o %t.per_module.files.table
; RUN: FileCheck %s -input-file=%t.per_module.files_0.prop -implicit-check-not="NotExported" --check-prefix=CHECK-KERNELONLY-PROP
; RUN: FileCheck %s -input-file=%t.per_module.files_1.prop -implicit-check-not="NotExported" --check-prefix=CHECK-PERMODULE-0-PROP
; RUN: FileCheck %s -input-file=%t.per_module.files_2.prop -implicit-check-not="NotExported" --check-prefix=CHECK-PERMODULE-2-PROP
;
; Per-kernel split
; RUN: sycl-post-link -properties -symbols -split=kernel -emit-exported-symbols -S < %s -o %t.per_kernel.files.table
; RUN: FileCheck %s -input-file=%t.per_kernel.files_0.prop --implicit-check-not="NotExported" --check-prefix=CHECK-KERNELONLY-PROP
; RUN: FileCheck %s -input-file=%t.per_kernel.files_1.prop --implicit-check-not="NotExported" --check-prefix=CHECK-KERNELONLY-PROP
; RUN: FileCheck %s -input-file=%t.per_kernel.files_2.prop --implicit-check-not="NotExported" --check-prefix=CHECK-PERKERNEL-0-PROP
; RUN: FileCheck %s -input-file=%t.per_kernel.files_3.prop --implicit-check-not="NotExported" --check-prefix=CHECK-PERKERNEL-1-PROP
; RUN: FileCheck %s -input-file=%t.per_kernel.files_4.prop --implicit-check-not="NotExported" --check-prefix=CHECK-PERKERNEL-2-PROP

target triple = "spir64-unknown-unknown"

define dso_local spir_kernel void @NotExportedSpirKernel1(float %arg1) #0 {
entry:
  ret void
}

define dso_local spir_kernel void @NotExportedSpirKernel2(float %arg1) #2 {
entry:
  ret void
}

define dso_local spir_func void @ExportedSpirFunc1(float %arg1) #0 {
entry:
  ret void
}

define dso_local spir_func void @ExportedSpirFunc2(i32 %arg1, i32 %arg2) #1 {
entry:
  ret void
}

define dso_local spir_func void @ExportedSpirFunc3(float %arg1) #0 {
entry:
  ret void
}

define dso_local spir_func void @NotExportedSpirFunc1(float %arg1) {
entry:
  ret void
}

attributes #0 = { "sycl-module-id"="a.cpp" }
attributes #1 = { "sycl-module-id"="b.cpp" }
attributes #2 = { "sycl-module-id"="c.cpp" }

; Global scope
; CHECK-GLOBAL-PROP: [SYCL/exported symbols]
; CHECK-GLOBAL-PROP-NEXT: ExportedSpirFunc1
; CHECK-GLOBAL-PROP-NEXT: ExportedSpirFunc2
; CHECK-GLOBAL-PROP-NEXT: ExportedSpirFunc3

; Per-module split
; CHECK-PERMODULE-2-PROP: [SYCL/exported symbols]
; CHECK-PERMODULE-2-PROP-NEXT: ExportedSpirFunc1
; CHECK-PERMODULE-2-PROP-NEXT: ExportedSpirFunc3
; CHECK-PERMODULE-2-PROP-NOT: ExportedSpirFunc2

; CHECK-PERMODULE-0-PROP: [SYCL/exported symbols]
; CHECK-PERMODULE-0-PROP-NEXT: ExportedSpirFunc2
; CHECK-PERMODULE-0-PROP-NOT: ExportedSpirFunc1
; CHECK-PERMODULE-0-PROP-NOT: ExportedSpirFunc3

; Per-kernel split
; CHECK-PERKERNEL-2-PROP: [SYCL/exported symbols]
; CHECK-PERKERNEL-2-PROP-NEXT: ExportedSpirFunc1
; CHECK-PERKERNEL-2-PROP-NOT: ExportedSpirFunc2
; CHECK-PERKERNEL-2-PROP-NOT: ExportedSpirFunc3

; CHECK-PERKERNEL-1-PROP: [SYCL/exported symbols]
; CHECK-PERKERNEL-1-PROP-NEXT: ExportedSpirFunc2
; CHECK-PERKERNEL-1-PROP-NOT: ExportedSpirFunc1
; CHECK-PERKERNEL-1-PROP-NOT: ExportedSpirFunc3

; CHECK-PERKERNEL-0-PROP: [SYCL/exported symbols]
; CHECK-PERKERNEL-0-PROP-NEXT: ExportedSpirFunc3
; CHECK-PERKERNEL-0-PROP-NOT: ExportedSpirFunc1
; CHECK-PERKERNEL-0-PROP-NOT: ExportedSpirFunc2

; Kernel-only generated modules should have no exported Symbols
; CHECK-KERNELONLY-PROP-NOT: [SYCL/exported symbols]
; CHECK-KERNELONLY-PROP-NOT: ExportedSpirFunc3
; CHECK-KERNELONLY-PROP-NOT: ExportedSpirFunc1
; CHECK-KERNELONLY-PROP-NOT: ExportedSpirFunc2
