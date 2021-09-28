; This test checks that the post-link tool generates list of exported symbols.
;
; Global scope
; RUN: sycl-post-link -symbols -emit-exported-symbols -S %s -o %t.global.files.table
; RUN: FileCheck %s -input-file=%t.global.files_0.prop --check-prefixes CHECK-GLOBAL-PROP
;
; Per-module split
; RUN: sycl-post-link -symbols -split=source -emit-exported-symbols -S %s -o %t.per_module.files.table
; RUN: FileCheck %s -input-file=%t.per_module.files_0.prop --check-prefixes CHECK-PERMODULE-0-PROP
; RUN: FileCheck %s -input-file=%t.per_module.files_1.prop --check-prefixes CHECK-PERMODULE-1-PROP
; RUN: FileCheck %s -input-file=%t.per_module.files_2.prop --check-prefixes CHECK-KERNELONLY-PROP
;
; Per-kernel split
; RUN: sycl-post-link -symbols -split=kernel -emit-exported-symbols -S %s -o %t.per_kernel.files.table
; RUN: FileCheck %s -input-file=%t.per_kernel.files_0.prop --check-prefixes CHECK-PERKERNEL-0-PROP
; RUN: FileCheck %s -input-file=%t.per_kernel.files_1.prop --check-prefixes CHECK-PERKERNEL-1-PROP
; RUN: FileCheck %s -input-file=%t.per_kernel.files_2.prop --check-prefixes CHECK-PERKERNEL-2-PROP
; RUN: FileCheck %s -input-file=%t.per_kernel.files_3.prop --check-prefixes CHECK-KERNELONLY-PROP
; RUN: FileCheck %s -input-file=%t.per_kernel.files_4.prop --check-prefixes CHECK-KERNELONLY-PROP

target triple = "spir64-unknown-unknown"

define dso_local spir_kernel void @SpirKernel1(float %arg1) #0 {
entry:
  ret void
}

define dso_local spir_kernel void @SpirKernel2(float %arg1) #2 {
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
; CHECK-GLOBAL-PROP-NOT: SpirKernel1
; CHECK-GLOBAL-PROP-NOT: NotExportedSpirFunc1

; Per-module split
; CHECK-PERMODULE-0-PROP: [SYCL/exported symbols]
; CHECK-PERMODULE-0-PROP-NEXT: ExportedSpirFunc1
; CHECK-PERMODULE-0-PROP-NEXT: ExportedSpirFunc3
; CHECK-PERMODULE-0-PROP-NOT: ExportedSpirFunc2
; CHECK-PERMODULE-0-PROP-NOT: SpirKernel1
; CHECK-PERMODULE-0-PROP-NOT: NotExportedSpirFunc1

; CHECK-PERMODULE-1-PROP: [SYCL/exported symbols]
; CHECK-PERMODULE-1-PROP-NEXT: ExportedSpirFunc2
; CHECK-PERMODULE-1-PROP-NOT: ExportedSpirFunc1
; CHECK-PERMODULE-1-PROP-NOT: ExportedSpirFunc3
; CHECK-PERMODULE-1-PROP-NOT: SpirKernel1
; CHECK-PERMODULE-1-PROP-NOT: NotExportedSpirFunc1

; Per-kernel split
; CHECK-PERKERNEL-0-PROP: [SYCL/exported symbols]
; CHECK-PERKERNEL-0-PROP-NEXT: ExportedSpirFunc1
; CHECK-PERKERNEL-0-PROP-NOT: ExportedSpirFunc2
; CHECK-PERKERNEL-0-PROP-NOT: ExportedSpirFunc3
; CHECK-PERKERNEL-0-PROP-NOT: SpirKernel1
; CHECK-PERKERNEL-0-PROP-NOT: NotExportedSpirFunc1

; CHECK-PERKERNEL-1-PROP: [SYCL/exported symbols]
; CHECK-PERKERNEL-1-PROP-NEXT: ExportedSpirFunc2
; CHECK-PERKERNEL-1-PROP-NOT: ExportedSpirFunc1
; CHECK-PERKERNEL-1-PROP-NOT: ExportedSpirFunc3
; CHECK-PERKERNEL-1-PROP-NOT: SpirKernel1
; CHECK-PERKERNEL-1-PROP-NOT: NotExportedSpirFunc1

; CHECK-PERKERNEL-2-PROP: [SYCL/exported symbols]
; CHECK-PERKERNEL-2-PROP-NEXT: ExportedSpirFunc3
; CHECK-PERKERNEL-2-PROP-NOT: ExportedSpirFunc1
; CHECK-PERKERNEL-2-PROP-NOT: ExportedSpirFunc2
; CHECK-PERKERNEL-2-PROP-NOT: SpirKernel1
; CHECK-PERKERNEL-2-PROP-NOT: NotExportedSpirFunc1

; Kernel-only generated modules should have no exported Symbols
; CHECK-KERNELONLY-PROP-NOT: [SYCL/exported symbols]
; CHECK-KERNELONLY-PROP-NOT: ExportedSpirFunc3
; CHECK-KERNELONLY-PROP-NOT: ExportedSpirFunc1
; CHECK-KERNELONLY-PROP-NOT: ExportedSpirFunc2
; CHECK-KERNELONLY-PROP-NOT: SpirKernel1
; CHECK-KERNELONLY-PROP-NOT: NotExportedSpirFunc1
