; RUN: opt < %s -passes=asan -verify-each -asan-opt=0 -asan-instrumentation-with-call-threshold=0 -asan-stack=0 -asan-globals=0 -asan-constructor-kind=none -S | FileCheck %s
; RUN: opt < %s -passes=asan -verify-each -asan-opt=0 -asan-instrumentation-with-call-threshold=0 -asan-stack=0 -asan-globals=0 -asan-constructor-kind=none -asan-spir-shadow-bounds=1 -S | FileCheck %s

; GPU printf lowering recognizes constant byte-array initializers as strings.
; Keep this representation while adding redzones and registering shadow for
; accesses through callee arguments.
target datalayout = "e-i64:64-n8:16:32:64"
target triple = "spir64-unknown-unknown"

@format = internal addrspace(2) constant [4 x i8] c"%s\0A\00"
@literal = private unnamed_addr addrspace(1) constant [6 x i8] c"Hello\00", align 1
@empty = private unnamed_addr addrspace(1) constant [1 x i8] c"\00", align 1
@wide = private addrspace(1) constant [2 x i16] [i16 65, i16 0], align 2

; CHECK: @format = internal addrspace(2) constant [4 x i8] c"%s\0A\00"
; CHECK: @literal = private unnamed_addr addrspace(1) constant [32 x i8] c"Hello\00{{.*}}", align 32
; CHECK: @empty = private unnamed_addr addrspace(1) constant [32 x i8] zeroinitializer, align 32
; CHECK: @wide = private addrspace(1) constant { [2 x i16], [28 x i8] }
; CHECK: @__AsanDeviceGlobalMetadata_{{.*}} = local_unnamed_addr addrspace(1) global [3 x { i64, i64, i64 }]
; CHECK-SAME: { i64 6, i64 32, i64 ptrtoint (ptr addrspace(1) @literal to i64) }
; CHECK-SAME: { i64 1, i64 32, i64 ptrtoint (ptr addrspace(1) @empty to i64) }
; CHECK-SAME: { i64 4, i64 32, i64 ptrtoint (ptr addrspace(1) @wide to i64) }

declare spir_func i32 @_Z18__spirv_ocl_printfPU3AS2Kcz(ptr addrspace(2), ...)

define spir_func i8 @callee(ptr addrspace(1) %p, i64 %index) sanitize_address noinline {
; CHECK-LABEL: define spir_func i8 @callee(
; CHECK: call void @__asan_load1_as1(
; CHECK-NEXT: %value = load i8, ptr addrspace(1) %element
  %element = getelementptr inbounds i8, ptr addrspace(1) %p, i64 %index
  %value = load i8, ptr addrspace(1) %element, align 1
  ret i8 %value
}

define spir_func void @caller(i64 %index) sanitize_address {
; CHECK-LABEL: define spir_func void @caller(
; CHECK: call spir_func i32 (ptr addrspace(2), ...) @_Z18__spirv_ocl_printfPU3AS2Kcz(ptr addrspace(2) @format, ptr addrspace(4) addrspacecast (ptr addrspace(1) @literal to ptr addrspace(4)))
; CHECK: call spir_func i32 (ptr addrspace(2), ...) @_Z18__spirv_ocl_printfPU3AS2Kcz(ptr addrspace(2) @format, ptr addrspace(4) addrspacecast (ptr addrspace(1) @empty to ptr addrspace(4)))
; CHECK: call spir_func i8 @callee(ptr addrspace(1) @literal, i64 %index)
  %printed = call spir_func i32 (ptr addrspace(2), ...) @_Z18__spirv_ocl_printfPU3AS2Kcz(ptr addrspace(2) @format, ptr addrspace(4) addrspacecast (ptr addrspace(1) @literal to ptr addrspace(4)))
  %printed_empty = call spir_func i32 (ptr addrspace(2), ...) @_Z18__spirv_ocl_printfPU3AS2Kcz(ptr addrspace(2) @format, ptr addrspace(4) addrspacecast (ptr addrspace(1) @empty to ptr addrspace(4)))
  %value = call spir_func i8 @callee(ptr addrspace(1) @literal, i64 %index)
  ret void
}
