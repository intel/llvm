; RUN: opt < %s -passes=asan -verify-each -asan-opt=0 -asan-instrumentation-with-call-threshold=0 -asan-stack=0 -asan-globals=0 -asan-constructor-kind=none -S | FileCheck %s
; RUN: opt < %s -passes=asan -verify-each -asan-opt=0 -asan-instrumentation-with-call-threshold=0 -asan-stack=0 -asan-globals=0 -asan-constructor-kind=none -asan-spir-shadow-bounds=1 -S | FileCheck %s

; The payload of a non-image-scope device_global uses USM, but the global
; wrapper containing its USM pointer still needs shadow memory. At -O0, that
; pointer is loaded through a generic this argument in a separate get_ptr().
; Register the wrapper without changing the payload size used by SYCL runtime.

target datalayout = "e-i64:64-v16:16-v24:32-v32:32-v48:64-v96:128-v192:256-v256:256-v512:512-v1024:1024-n8:16:32:64"
target triple = "spir64-unknown-unknown"

@dev_global = addrspace(1) global { ptr addrspace(1), [5 x i8] } zeroinitializer #0
@dev_global_false = addrspace(1) global { ptr addrspace(1), [5 x i8] } zeroinitializer #1

; CHECK: @dev_global = addrspace(1) global { { ptr addrspace(1), [5 x i8] }, [16 x i8] } zeroinitializer, align 32 [[NO_SCOPE:#[0-9]+]]
; CHECK: @dev_global_false = addrspace(1) global { { ptr addrspace(1), [5 x i8] }, [16 x i8] } zeroinitializer, align 32 [[FALSE_SCOPE:#[0-9]+]]
; CHECK: @__AsanDeviceGlobalMetadata_{{.*}} = local_unnamed_addr addrspace(1) global [2 x { i64, i64, i64 }]
; CHECK-SAME: { i64 16, i64 32, i64 ptrtoint (ptr addrspace(1) @dev_global to i64) }
; CHECK-SAME: { i64 16, i64 32, i64 ptrtoint (ptr addrspace(1) @dev_global_false to i64) }

define spir_func ptr addrspace(1) @get_usm_pointer(ptr addrspace(4) %this) sanitize_address noinline {
; CHECK-LABEL: define spir_func ptr addrspace(1) @get_usm_pointer(
; CHECK: call void @__asan_load8_as4(
; CHECK-NEXT: %p = load ptr addrspace(1), ptr addrspace(4) %usmptr
  %usmptr = getelementptr inbounds { ptr addrspace(1), [5 x i8] }, ptr addrspace(4) %this, i32 0, i32 0
  %p = load ptr addrspace(1), ptr addrspace(4) %usmptr, align 8
  ret ptr addrspace(1) %p
}

define spir_func void @access_usm_data() sanitize_address {
; CHECK-LABEL: define spir_func void @access_usm_data(
; CHECK: call void @__asan_load8_as1(
; CHECK-NEXT: %direct = load ptr addrspace(1), ptr addrspace(1) @dev_global
; CHECK: %p = call spir_func ptr addrspace(1) @get_usm_pointer(ptr addrspace(4) addrspacecast (ptr addrspace(1) @dev_global to ptr addrspace(4)))
; CHECK: call void @__asan_store1_as1(
; CHECK-NEXT: store i8 42, ptr addrspace(1) %element
  %direct = load ptr addrspace(1), ptr addrspace(1) @dev_global, align 8
  %p = call spir_func ptr addrspace(1) @get_usm_pointer(ptr addrspace(4) addrspacecast (ptr addrspace(1) @dev_global to ptr addrspace(4)))
  %element = getelementptr inbounds i8, ptr addrspace(1) %p, i64 8
  store i8 42, ptr addrspace(1) %element, align 1
  ret void
}

; CHECK: attributes [[NO_SCOPE]] = { "sycl-device-global-size"="5" "sycl-unique-id"="dev_global" }
; CHECK: attributes [[FALSE_SCOPE]] = { "sycl-device-global-size"="5" "sycl-device-image-scope"="false" "sycl-unique-id"="dev_global_false" }

attributes #0 = { "sycl-device-global-size"="5" "sycl-unique-id"="dev_global" }
attributes #1 = { "sycl-device-global-size"="5" "sycl-device-image-scope"="false" "sycl-unique-id"="dev_global_false" }
