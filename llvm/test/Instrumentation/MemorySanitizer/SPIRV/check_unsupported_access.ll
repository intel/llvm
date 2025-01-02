; RUN: opt < %s -passes=msan -msan-instrumentation-with-call-threshold=0 -msan-eager-checks=1 -S | FileCheck %s
target datalayout = "e-i64:64-v16:16-v24:32-v32:32-v48:64-v96:128-v192:256-v256:256-v512:512-v1024:1024-n8:16:32:64-G1"
target triple = "spir64-unknown-unknown"

define spir_kernel void @MyKernel(ptr %__SYCLKernel) {
; CHECK-LABEL: @MyKernel
entry:
  %_arg_array1.addr = alloca ptr addrspace(1), i32 0, align 8
; CHECK: %_arg_array1.addr = alloca ptr addrspace(1){{.*!nosanitize}}
  %_arg_array1.addr.ascast = addrspacecast ptr %_arg_array1.addr to ptr addrspace(4)
; CHECK: %_arg_array1.addr.ascast = addrspacecast ptr %_arg_array1.addr to ptr addrspace(4){{.*!nosanitize}}
  call void @llvm.lifetime.start.p0(i64 64, ptr %__SYCLKernel)
; CHECK: call void @llvm.lifetime.start.p0(i64 64, ptr %__SYCLKernel){{.*!nosanitize}}
  ret void
}

; Function Attrs: nocallback nofree nosync nounwind willreturn memory(argmem: readwrite)
declare void @llvm.lifetime.start.p0(i64 immarg, ptr nocapture) #0

attributes #0 = { nocallback nofree nosync nounwind willreturn memory(argmem: readwrite) }

