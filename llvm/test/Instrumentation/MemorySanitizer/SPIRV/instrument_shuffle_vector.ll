; RUN: opt < %s -passes=msan -msan-instrumentation-with-call-threshold=0 -msan-eager-checks=1 -S | FileCheck %s
target datalayout = "e-i64:64-v16:16-v24:32-v32:32-v48:64-v96:128-v192:256-v256:256-v512:512-v1024:1024-n8:16:32:64-G1"
target triple = "spir64-unknown-unknown"

define spir_kernel void @MyKernel(ptr addrspace(1) noundef align 4 %_arg_array) sanitize_memory {
entry:
  %arrayidx3 = getelementptr inbounds <3 x i16>, ptr addrspace(1) %_arg_array, i64 0
  %extractVec4 = shufflevector <3 x i16> <i16 0, i16 0, i16 0>, <3 x i16> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 poison>
; CHECK: [[REG1:%[0-9]+]] = call i64 @__msan_get_shadow
; CHECK: [[REG2:%[0-9]+]] = inttoptr i64 [[REG1]] to ptr addrspace(1)
; CHECK: store <4 x i16> zeroinitializer, ptr addrspace(1) [[REG2]]
  store <4 x i16> %extractVec4, ptr addrspace(1) %arrayidx3, align 8
  br label %exit

exit:
  ret void
}
