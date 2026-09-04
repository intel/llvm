; This test simulates two device images that will be dynamically linked
; together (see sycl/doc/design/SharedLibraries.md): each image carries its
; own ASan kernel-metadata global, already stamped by the instrumentation pass
; with a distinct per-module id (the "_aaa..." / "_bbb..." suffix). It checks
; that:
;  1. sycl-post-link keeps both `__AsanKernelMetadata_<id>` globals -- with
;     their own id preserved -- in every split device image, not just the one
;     that defines the corresponding kernel.
;  2. For each id present in the module, a matching `__sanitizerModule_<id>`
;     sentinel kernel is emitted in every split device image, so the UR
;     sanitizer layer can look up the metadata global for each image
;     independently, without id collisions across the dynamically-linked
;     images.
;
; RUN: sycl-post-link -properties -split=source -symbols -S < %s -o %t.table
; RUN: FileCheck %s -input-file=%t_0.ll --check-prefixes CHECK-MOD0,CHECK
; RUN: FileCheck %s -input-file=%t_1.ll --check-prefixes CHECK-MOD1,CHECK

target datalayout = "e-i64:64-v16:16-v24:32-v32:32-v48:64-v96:128-v192:256-v256:256-v512:512-v1024:1024-n8:16:32:64"
target triple = "spir64-unknown-unknown"

; Both metadata globals -- for both images -- keep their own id.
; CHECK-DAG: @__AsanKernelMetadata_aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa = {{.*}}global
; CHECK-DAG: @__AsanKernelMetadata_bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb = {{.*}}global

@__asan_kernel_a = internal addrspace(1) constant [8 x i8] c"KernelA\00"
@__AsanKernelMetadata_aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa = appending dso_local local_unnamed_addr addrspace(1) global [1 x { i64, i64 }] [{ i64, i64 } { i64 ptrtoint (ptr addrspace(1) @__asan_kernel_a to i64), i64 7 }] #2

; CHECK-MOD1: define weak_odr dso_local spir_kernel void @KernelA()
; CHECK-MOD0-NOT: define weak_odr dso_local spir_kernel void @KernelA()
define weak_odr dso_local spir_kernel void @KernelA() #0 {
entry:
  ret void
}

@__asan_kernel_b = internal addrspace(1) constant [8 x i8] c"KernelB\00"
@__AsanKernelMetadata_bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb = appending dso_local local_unnamed_addr addrspace(1) global [1 x { i64, i64 }] [{ i64, i64 } { i64 ptrtoint (ptr addrspace(1) @__asan_kernel_b to i64), i64 7 }] #3

; CHECK-MOD0: define weak_odr dso_local spir_kernel void @KernelB()
; CHECK-MOD1-NOT: define weak_odr dso_local spir_kernel void @KernelB()
define weak_odr dso_local spir_kernel void @KernelB() #1 {
entry:
  ret void
}

; Both split images must carry both sentinel kernels: one per id, regardless
; of which image actually defines the corresponding user kernel.
; CHECK-DAG: define weak_odr dso_local spir_kernel void @__sanitizerModule_aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa()
; CHECK-DAG: define weak_odr dso_local spir_kernel void @__sanitizerModule_bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb()

; CHECK-DAG: "sycl-unique-id"="__AsanKernelMetadata_aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"
; CHECK-DAG: "sycl-unique-id"="__AsanKernelMetadata_bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb"
attributes #0 = { sanitize_address "sycl-module-id"="a.cpp" }
attributes #1 = { sanitize_address "sycl-module-id"="b.cpp" }
attributes #2 = { "sycl-device-global-size"="16" "sycl-device-image-scope" "sycl-host-access"="0" "sycl-unique-id"="__AsanKernelMetadata_aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa" }
attributes #3 = { "sycl-device-global-size"="16" "sycl-device-image-scope" "sycl-host-access"="0" "sycl-unique-id"="__AsanKernelMetadata_bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb" }
