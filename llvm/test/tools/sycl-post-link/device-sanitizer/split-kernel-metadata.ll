; This test checks that after splitting, the sanitizer kernel metadata of each
; device image only describes the kernels defined in that image: the entries of
; the other kernels are zeroed out, and their kernel name strings are not copied
; into the image. Otherwise every image carries the names of all kernels of the
; program, and with -split=kernel the total size of the device images grows
; quadratically with the number of kernels. The number of entries is kept, as
; the metadata is a device global and all device images must agree on its size.

; RUN: split-file %s %t

; RUN: sycl-post-link -properties -device-globals -split=kernel -symbols -S < %t/asan.ll -o %t/asan.table
; RUN: FileCheck %s -input-file=%t/asan_0.ll --check-prefix ASAN-B --implicit-check-not=KernelA
; RUN: FileCheck %s -input-file=%t/asan_1.ll --check-prefix ASAN-A --implicit-check-not=KernelB
; RUN: FileCheck %s -input-file=%t/asan_0.prop --check-prefix ASAN-PROP
; RUN: FileCheck %s -input-file=%t/asan_1.prop --check-prefix ASAN-PROP

; RUN: sycl-post-link -properties -split=source -symbols -S < %t/asan.ll -o %t/asan-nosplit.table
; RUN: FileCheck %s -input-file=%t/asan-nosplit_0.ll --check-prefix ASAN-NOSPLIT

; RUN: sycl-post-link -properties -split=kernel -symbols -S < %t/msan.ll -o %t/msan.table
; RUN: FileCheck %s -input-file=%t/msan_0.ll --check-prefix MSAN-B --implicit-check-not=KernelA
; RUN: FileCheck %s -input-file=%t/msan_1.ll --check-prefix MSAN-A --implicit-check-not=KernelB

; RUN: sycl-post-link -properties -split=kernel -symbols -S < %t/tsan.ll -o %t/tsan.table
; RUN: FileCheck %s -input-file=%t/tsan_0.ll --check-prefix TSAN-B --implicit-check-not=KernelA
; RUN: FileCheck %s -input-file=%t/tsan_1.ll --check-prefix TSAN-A --implicit-check-not=KernelB

; Kernels from two modules with different ids: all the entries of the metadata
; global of the other module are zeroed out, and its sentinel kernel is kept.
; RUN: sycl-post-link -properties -device-globals -split=kernel -symbols -S < %t/two-ids.ll -o %t/two-ids.table
; RUN: FileCheck %s -input-file=%t/two-ids_0.ll --check-prefix TWO-IDS-B --implicit-check-not=KernelA
; RUN: FileCheck %s -input-file=%t/two-ids_1.ll --check-prefix TWO-IDS-A --implicit-check-not=KernelB
; RUN: FileCheck %s -input-file=%t/two-ids_0.prop --check-prefix TWO-IDS-PROP
; RUN: FileCheck %s -input-file=%t/two-ids_1.prop --check-prefix TWO-IDS-PROP

; ASAN-A: @__asan_kernel = internal addrspace(2) constant [8 x i8] c"KernelA\00"
; ASAN-A: @__AsanKernelMetadata_aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa = dso_local local_unnamed_addr addrspace(1) global %{{.*}} { [2 x { i64, i64, i64 }] [{ i64, i64, i64 } { i64 ptrtoint (ptr addrspace(2) @__asan_kernel to i64), i64 7, i64 12 }, { i64, i64, i64 } zeroinitializer] }, !spirv.Decorations ![[#]] #[[#ATTR:]]
; ASAN-A: define weak_odr dso_local spir_kernel void @KernelA()
; ASAN-A: attributes #[[#ATTR]] = { "sycl-device-global-size"="48" "sycl-device-image-scope" "sycl-host-access"="0" "sycl-unique-id"="__AsanKernelMetadata_aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa" }

; ASAN-B: @__asan_kernel.1 = internal addrspace(2) constant [8 x i8] c"KernelB\00"
; ASAN-B: @__AsanKernelMetadata_aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa = dso_local local_unnamed_addr addrspace(1) global %{{.*}} { [2 x { i64, i64, i64 }] [{ i64, i64, i64 } zeroinitializer, { i64, i64, i64 } { i64 ptrtoint (ptr addrspace(2) @__asan_kernel.1 to i64), i64 7, i64 12 }] }, !spirv.Decorations ![[#]] #[[#ATTR:]]
; ASAN-B: define weak_odr dso_local spir_kernel void @KernelB()
; ASAN-B: attributes #[[#ATTR]] = { "sycl-device-global-size"="48" {{.*}} }

; The property value encodes the size of the device global, so it must be the
; same in every device image.
; ASAN-PROP: [SYCL/device globals]
; ASAN-PROP-NEXT: __AsanKernelMetadata_aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa=2|ABAAAAAAAAAMAAAABAAAAA

; ASAN-NOSPLIT: @__asan_kernel = internal addrspace(2) constant [8 x i8] c"KernelA\00"
; ASAN-NOSPLIT: @__asan_kernel.1 = internal addrspace(2) constant [8 x i8] c"KernelB\00"
; ASAN-NOSPLIT: @__AsanKernelMetadata_aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa = {{.*}} { [2 x { i64, i64, i64 }] [{ i64, i64, i64 } { i64 ptrtoint (ptr addrspace(2) @__asan_kernel to i64), i64 7, i64 12 }, { i64, i64, i64 } { i64 ptrtoint (ptr addrspace(2) @__asan_kernel.1 to i64), i64 7, i64 12 }] }, !spirv.Decorations ![[#]] #[[#ATTR:]]
; ASAN-NOSPLIT: attributes #[[#ATTR]] = { "sycl-device-global-size"="48" {{.*}} }

; MSAN-A: @__msan_kernel = internal addrspace(2) constant [8 x i8] c"KernelA\00"
; MSAN-A: @__MsanKernelMetadata_aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa = {{.*}} { [2 x { i64, i64, i64 }] [{ i64, i64, i64 } { i64 ptrtoint (ptr addrspace(2) @__msan_kernel to i64), i64 7, i64 0 }, { i64, i64, i64 } zeroinitializer] }, !spirv.Decorations ![[#]] #[[#ATTR:]]
; MSAN-A: attributes #[[#ATTR]] = { "sycl-device-global-size"="48" {{.*}} }

; MSAN-B: @__msan_kernel.1 = internal addrspace(2) constant [8 x i8] c"KernelB\00"
; MSAN-B: @__MsanKernelMetadata_aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa = {{.*}} { [2 x { i64, i64, i64 }] [{ i64, i64, i64 } zeroinitializer, { i64, i64, i64 } { i64 ptrtoint (ptr addrspace(2) @__msan_kernel.1 to i64), i64 7, i64 0 }] }, !spirv.Decorations ![[#]] #[[#ATTR:]]
; MSAN-B: attributes #[[#ATTR]] = { "sycl-device-global-size"="48" {{.*}} }

; TSAN-A: @__tsan_kernel = internal addrspace(2) constant [8 x i8] c"KernelA\00"
; TSAN-A: @__TsanKernelMetadata_aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa = {{.*}} { [2 x { i64, i64 }] [{ i64, i64 } { i64 ptrtoint (ptr addrspace(2) @__tsan_kernel to i64), i64 7 }, { i64, i64 } zeroinitializer] }, !spirv.Decorations ![[#]] #[[#ATTR:]]
; TSAN-A: attributes #[[#ATTR]] = { "sycl-device-global-size"="32" {{.*}} }

; TSAN-B: @__tsan_kernel.1 = internal addrspace(2) constant [8 x i8] c"KernelB\00"
; TSAN-B: @__TsanKernelMetadata_aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa = {{.*}} { [2 x { i64, i64 }] [{ i64, i64 } zeroinitializer, { i64, i64 } { i64 ptrtoint (ptr addrspace(2) @__tsan_kernel.1 to i64), i64 7 }] }, !spirv.Decorations ![[#]] #[[#ATTR:]]
; TSAN-B: attributes #[[#ATTR]] = { "sycl-device-global-size"="32" {{.*}} }

; TWO-IDS-A: @__AsanKernelMetadata_aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa = {{.*}} { [1 x { i64, i64, i64 }] [{ i64, i64, i64 } { i64 ptrtoint (ptr addrspace(2) @__asan_kernel_a to i64), i64 7, i64 12 }] }, !spirv.Decorations ![[#]] #[[#ATTR_A:]]
; TWO-IDS-A: @__AsanKernelMetadata_bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb = {{.*}} global %{{.*}} zeroinitializer, !spirv.Decorations ![[#]] #[[#ATTR_B:]]
; TWO-IDS-A-DAG: define weak_odr dso_local spir_kernel void @KernelA()
; TWO-IDS-A-DAG: define weak_odr dso_local spir_kernel void @__sanitizerModule_aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa()
; TWO-IDS-A-DAG: define weak_odr dso_local spir_kernel void @__sanitizerModule_bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb()
; TWO-IDS-A: attributes #[[#ATTR_A]] = { "sycl-device-global-size"="24" {{.*}} }
; TWO-IDS-A: attributes #[[#ATTR_B]] = { "sycl-device-global-size"="24" {{.*}} }

; TWO-IDS-B: @__AsanKernelMetadata_aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa = {{.*}} global %{{.*}} zeroinitializer, !spirv.Decorations ![[#]] #[[#ATTR_A:]]
; TWO-IDS-B: @__AsanKernelMetadata_bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb = {{.*}} { [1 x { i64, i64, i64 }] [{ i64, i64, i64 } { i64 ptrtoint (ptr addrspace(2) @__asan_kernel_b to i64), i64 7, i64 12 }] }, !spirv.Decorations ![[#]] #[[#ATTR_B:]]
; TWO-IDS-B-DAG: define weak_odr dso_local spir_kernel void @KernelB()
; TWO-IDS-B-DAG: define weak_odr dso_local spir_kernel void @__sanitizerModule_aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa()
; TWO-IDS-B-DAG: define weak_odr dso_local spir_kernel void @__sanitizerModule_bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb()
; TWO-IDS-B: attributes #[[#ATTR_A]] = { "sycl-device-global-size"="24" {{.*}} }
; TWO-IDS-B: attributes #[[#ATTR_B]] = { "sycl-device-global-size"="24" {{.*}} }

; TWO-IDS-PROP: [SYCL/device globals]
; TWO-IDS-PROP-NEXT: __AsanKernelMetadata_aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa=2|ABAAAAAAAAAGAAAABAAAAA
; TWO-IDS-PROP-NEXT: __AsanKernelMetadata_bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb=2|ABAAAAAAAAAGAAAABAAAAA

;--- asan.ll
target datalayout = "e-i64:64-v16:16-v24:32-v32:32-v48:64-v96:128-v192:256-v256:256-v512:512-v1024:1024-n8:16:32:64"
target triple = "spir64-unknown-unknown"

@__asan_kernel = internal addrspace(2) constant [8 x i8] c"KernelA\00"
@__asan_kernel.1 = internal addrspace(2) constant [8 x i8] c"KernelB\00"
@__AsanKernelMetadata_aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa = dso_local local_unnamed_addr addrspace(1) global [2 x { i64, i64, i64 }] [{ i64, i64, i64 } { i64 ptrtoint (ptr addrspace(2) @__asan_kernel to i64), i64 7, i64 12 }, { i64, i64, i64 } { i64 ptrtoint (ptr addrspace(2) @__asan_kernel.1 to i64), i64 7, i64 12 }] #1

define weak_odr dso_local spir_kernel void @KernelA() #0 {
entry:
  ret void
}

define weak_odr dso_local spir_kernel void @KernelB() #0 {
entry:
  ret void
}

attributes #0 = { sanitize_address "sycl-module-id"="a.cpp" }
attributes #1 = { "sycl-device-global-size"="48" "sycl-device-image-scope" "sycl-host-access"="0" "sycl-unique-id"="__AsanKernelMetadata_aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa" }

;--- msan.ll
target datalayout = "e-i64:64-v16:16-v24:32-v32:32-v48:64-v96:128-v192:256-v256:256-v512:512-v1024:1024-n8:16:32:64"
target triple = "spir64-unknown-unknown"

@__msan_kernel = internal addrspace(2) constant [8 x i8] c"KernelA\00"
@__msan_kernel.1 = internal addrspace(2) constant [8 x i8] c"KernelB\00"
@__MsanKernelMetadata_aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa = dso_local local_unnamed_addr addrspace(1) global [2 x { i64, i64, i64 }] [{ i64, i64, i64 } { i64 ptrtoint (ptr addrspace(2) @__msan_kernel to i64), i64 7, i64 0 }, { i64, i64, i64 } { i64 ptrtoint (ptr addrspace(2) @__msan_kernel.1 to i64), i64 7, i64 0 }] #1

define weak_odr dso_local spir_kernel void @KernelA() #0 {
entry:
  ret void
}

define weak_odr dso_local spir_kernel void @KernelB() #0 {
entry:
  ret void
}

attributes #0 = { sanitize_memory "sycl-module-id"="a.cpp" }
attributes #1 = { "sycl-device-global-size"="48" "sycl-device-image-scope" "sycl-host-access"="0" "sycl-unique-id"="__MsanKernelMetadata_aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa" }

;--- tsan.ll
target datalayout = "e-i64:64-v16:16-v24:32-v32:32-v48:64-v96:128-v192:256-v256:256-v512:512-v1024:1024-n8:16:32:64"
target triple = "spir64-unknown-unknown"

@__tsan_kernel = internal addrspace(2) constant [8 x i8] c"KernelA\00"
@__tsan_kernel.1 = internal addrspace(2) constant [8 x i8] c"KernelB\00"
@__TsanKernelMetadata_aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa = dso_local local_unnamed_addr addrspace(1) global [2 x { i64, i64 }] [{ i64, i64 } { i64 ptrtoint (ptr addrspace(2) @__tsan_kernel to i64), i64 7 }, { i64, i64 } { i64 ptrtoint (ptr addrspace(2) @__tsan_kernel.1 to i64), i64 7 }] #1

define weak_odr dso_local spir_kernel void @KernelA() #0 {
entry:
  ret void
}

define weak_odr dso_local spir_kernel void @KernelB() #0 {
entry:
  ret void
}

attributes #0 = { sanitize_thread "sycl-module-id"="a.cpp" }
attributes #1 = { "sycl-device-global-size"="32" "sycl-device-image-scope" "sycl-host-access"="0" "sycl-unique-id"="__TsanKernelMetadata_aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa" }

;--- two-ids.ll
target datalayout = "e-i64:64-v16:16-v24:32-v32:32-v48:64-v96:128-v192:256-v256:256-v512:512-v1024:1024-n8:16:32:64"
target triple = "spir64-unknown-unknown"

@__asan_kernel_a = internal addrspace(2) constant [8 x i8] c"KernelA\00"
@__AsanKernelMetadata_aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa = dso_local local_unnamed_addr addrspace(1) global [1 x { i64, i64, i64 }] [{ i64, i64, i64 } { i64 ptrtoint (ptr addrspace(2) @__asan_kernel_a to i64), i64 7, i64 12 }] #2

@__asan_kernel_b = internal addrspace(2) constant [8 x i8] c"KernelB\00"
@__AsanKernelMetadata_bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb = dso_local local_unnamed_addr addrspace(1) global [1 x { i64, i64, i64 }] [{ i64, i64, i64 } { i64 ptrtoint (ptr addrspace(2) @__asan_kernel_b to i64), i64 7, i64 12 }] #3

define weak_odr dso_local spir_kernel void @KernelA() #0 {
entry:
  ret void
}

define weak_odr dso_local spir_kernel void @KernelB() #1 {
entry:
  ret void
}

attributes #0 = { sanitize_address "sycl-module-id"="a.cpp" }
attributes #1 = { sanitize_address "sycl-module-id"="b.cpp" }
attributes #2 = { "sycl-device-global-size"="24" "sycl-device-image-scope" "sycl-host-access"="0" "sycl-unique-id"="__AsanKernelMetadata_aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa" }
attributes #3 = { "sycl-device-global-size"="24" "sycl-device-image-scope" "sycl-host-access"="0" "sycl-unique-id"="__AsanKernelMetadata_bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb" }