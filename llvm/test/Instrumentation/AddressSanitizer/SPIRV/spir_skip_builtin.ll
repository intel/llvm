; RUN: opt < %s -passes=asan -asan-instrumentation-with-call-threshold=0 -asan-stack=0 -asan-globals=0 -S | FileCheck %s

target datalayout = "e-i64:64-v16:16-v24:32-v32:32-v48:64-v96:128-v192:256-v256:256-v512:512-v1024:1024-n8:16:32:64"
target triple = "spir64-unknown-unknown"

@__spirv_BuiltInGlobalInvocationId = external addrspace(1) constant <3 x i64>

; Function Attrs: sanitize_address
define spir_kernel void @_ZTSN4sycl3_V16detail19__pf_kernel_wrapperIZZ4mainENKUlRNS0_7handlerEE_clES4_E19bufferByRange2_initEE() #0 {
entry:
  ; CHECK-NOT: {{ __asan_load8.*__spirv_BuiltInGlobalInvocationId }}
  %0 = load i64, ptr addrspace(1) getelementptr inbounds (i8, ptr addrspace(1) @__spirv_BuiltInGlobalInvocationId, i64 8), align 8
  ret void
}

attributes #0 = { sanitize_address }

