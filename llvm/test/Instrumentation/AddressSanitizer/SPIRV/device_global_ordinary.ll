; RUN: opt < %s -passes=asan -verify-each -asan-opt=0 -asan-instrumentation-with-call-threshold=0 -asan-stack=0 -asan-globals=0 -asan-constructor-kind=none -S | FileCheck %s
; RUN: opt < %s -passes=asan -verify-each -asan-opt=0 -asan-instrumentation-with-call-threshold=0 -asan-stack=0 -asan-globals=0 -asan-constructor-kind=none -asan-spir-shadow-bounds=1 -S | FileCheck %s

; Ordinary device globals need shadow registration even when accesses through
; callee arguments cannot be traced back to the global. Constants in global
; address space need registration too, and explicit alignment must be preserved.
target datalayout = "e-i64:64-n8:16:32:64"
target triple = "spir64-unknown-unknown"

@ordinary_global = internal addrspace(1) global [4 x i32] zeroinitializer, align 4
@aligned_global = addrspace(1) global i32 0, align 64
@constant_global = private addrspace(1) constant i32 1, align 4

; Declarations, other address spaces, compiler/runtime globals, and target
; extension types cannot use the ordinary device-global registration path.
@declaration = external addrspace(1) global i32
@constant_as = addrspace(2) constant i32 1
@private_as = global i32 0
@__AsanRuntimeGlobal = addrspace(1) global i32 0
@__spirv_BuiltInTest = addrspace(1) constant i32 0
@llvm.test_global = addrspace(1) global i32 0
@__usid_str = addrspace(1) constant [1 x i8] zeroinitializer
@__profd_test = addrspace(1) global i32 0
@__profc_test = addrspace(1) global i32 0
@channel = addrspace(1) global [4 x target("spirv.Channel")] zeroinitializer
@nosanitize = addrspace(1) global i32 0, no_sanitize_address
@imported_global = available_externally addrspace(1) constant i32 1

; CHECK: @ordinary_global = internal addrspace(1) global { [4 x i32], [16 x i8] } zeroinitializer, align 32
; CHECK: @aligned_global = addrspace(1) global { i32, [28 x i8] } zeroinitializer, align 64
; CHECK: @constant_global = private addrspace(1) constant { i32, [28 x i8] } { i32 1, [28 x i8] zeroinitializer }, align 32
; CHECK: @declaration = external addrspace(1) global i32
; CHECK: @constant_as = addrspace(2) constant i32 1
; CHECK: @private_as = global i32 0
; CHECK: @__AsanRuntimeGlobal = addrspace(1) global i32 0
; CHECK: @__spirv_BuiltInTest = addrspace(1) constant i32 0
; CHECK: @llvm.test_global = addrspace(1) global i32 0
; CHECK: @__usid_str = addrspace(1) constant [1 x i8] zeroinitializer
; CHECK: @__profd_test = addrspace(1) global i32 0
; CHECK: @__profc_test = addrspace(1) global i32 0
; CHECK: @channel = addrspace(1) global [4 x target("spirv.Channel")] zeroinitializer
; CHECK: @nosanitize = addrspace(1) global i32 0, no_sanitize_address
; CHECK: @imported_global = available_externally addrspace(1) constant i32 1
; CHECK: @__AsanDeviceGlobalMetadata_{{.*}} = local_unnamed_addr addrspace(1) global [3 x { i64, i64, i64 }]
; CHECK-SAME: { i64 16, i64 32, i64 ptrtoint (ptr addrspace(1) @ordinary_global to i64) }
; CHECK-SAME: { i64 4, i64 32, i64 ptrtoint (ptr addrspace(1) @aligned_global to i64) }
; CHECK-SAME: { i64 4, i64 32, i64 ptrtoint (ptr addrspace(1) @constant_global to i64) }

define spir_func i32 @callee(ptr addrspace(1) captures(none) %p) sanitize_address noinline {
; CHECK-LABEL: define spir_func i32 @callee(
; CHECK: call void @__asan_load4_as1(
; CHECK-NEXT: %value = load i32, ptr addrspace(1) %p
; CHECK: call void @__asan_store4_as1(
; CHECK-NEXT: store i32 1, ptr addrspace(1) %p
  %value = load i32, ptr addrspace(1) %p, align 4
  store i32 1, ptr addrspace(1) %p, align 4
  ret i32 %value
}

define spir_func i32 @generic_callee(ptr addrspace(4) captures(none) %p) sanitize_address noinline {
; CHECK-LABEL: define spir_func i32 @generic_callee(
; CHECK: call void @__asan_load4_as4(
; CHECK-NEXT: %value = load i32, ptr addrspace(4) %p
  %value = load i32, ptr addrspace(4) %p, align 4
  ret i32 %value
}

define spir_func void @caller(i64 %index) sanitize_address {
; CHECK-LABEL: define spir_func void @caller(
; CHECK: call void @__asan_store4_as1(
; CHECK-NEXT: store i32 2, ptr addrspace(1) %element
; CHECK: call spir_func i32 @callee(ptr addrspace(1) @ordinary_global)
; CHECK: %generic = addrspacecast ptr addrspace(1) @ordinary_global to ptr addrspace(4)
; CHECK-NEXT: call spir_func i32 @generic_callee(ptr addrspace(4) %generic)
  %element = getelementptr inbounds [4 x i32], ptr addrspace(1) @ordinary_global, i64 0, i64 %index
  store i32 2, ptr addrspace(1) %element, align 4
  %value = call spir_func i32 @callee(ptr addrspace(1) @ordinary_global)
  %generic = addrspacecast ptr addrspace(1) @ordinary_global to ptr addrspace(4)
  %generic_value = call spir_func i32 @generic_callee(ptr addrspace(4) %generic)
  ret void
}
