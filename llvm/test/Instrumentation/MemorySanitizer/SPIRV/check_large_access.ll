; RUN: opt < %s -passes=msan -msan-instrumentation-with-call-threshold=0 -msan-eager-checks=1 -S | FileCheck %s
;
; Verify that MSan reduces the shadow of a >64-bit vector to a scalar using
; llvm.vector.reduce.or rather than a direct bitcast (e.g. to i96), which is
; unsupported by SPIR-V.

target datalayout = "e-i64:64-v16:16-v24:32-v32:32-v48:64-v96:128-v192:256-v256:256-v512:512-v1024:1024-n8:16:32:64-G1"
target triple = "spir64-unknown-unknown"

; Function Attrs: sanitize_memory
define spir_kernel void @MyKernel(ptr addrspace(1) noundef align 4 %_arg_array) #0 {
; CHECK-LABEL: @MyKernel(
entry:
  %loadVecN = load <4 x i32>, ptr addrspace(1) %_arg_array, align 16
  %extractVec = shufflevector <4 x i32> %loadVecN, <4 x i32> zeroinitializer, <3 x i32> <i32 0, i32 1, i32 2>
  %div = sdiv <3 x i32> zeroinitializer, %extractVec
  ; CHECK: %[[SHADOW:.*]] = shufflevector <4 x i32>
  ; CHECK: call i32 @llvm.vector.reduce.or.v3i32(<3 x i32> %[[SHADOW]])
  ; CHECK-NOT: bitcast <3 x i32>
  ret void
}

attributes #0 = { sanitize_memory }
