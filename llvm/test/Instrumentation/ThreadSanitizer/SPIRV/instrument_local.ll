; RUN: opt < %s -passes='function(tsan),module(tsan-module)' -tsan-instrument-func-entry-exit=0 -tsan-instrument-memintrinsics=0 -S | FileCheck %s
target datalayout = "e-i64:64-v16:16-v24:32-v32:32-v48:64-v96:128-v192:256-v256:256-v512:512-v1024:1024-n8:16:32:64-G1"
target triple = "spir64-unknown-unknown"

@WGLocalMem.0 = external addrspace(3) global i32

define spir_func void @foo() #0 {
entry:
; CHECK-LABEL: define spir_func void @foo()
; CHECK: call void @__tsan_write4(i64 ptrtoint (ptr addrspace(3) @WGLocalMem.0 to i64), i32 3
  store i32 1, ptr addrspace(3) @WGLocalMem.0, align 4
  br label %exit

exit:
  ret void
}

define spir_kernel void @_ZTSZZ4mainENKUlRN4sycl3_V17handlerEE_clES2_E8MyKernel(ptr addrspace(3) noundef align 4 %_arg_acc) #0 {
entry:
; CHECK-LABEL: define spir_kernel void @_ZTSZZ4mainENKUlRN4sycl3_V17handlerEE_clES2_E8MyKernel(ptr addrspace(3) noundef align 4 %_arg_acc)
; CHECK: call void @__tsan_cleanup_static_local(i64 ptrtoint (ptr addrspace(3) @WGLocalMem.0 to i64), i64 4)
; CHECK: %local_args = alloca i64, align 8
; CHECK-NEXT: [[REG1:%[0-9]+]] = getelementptr i64, ptr %local_args, i32 0
; CHECK-NEXT: [[REG2:%[0-9]+]] = ptrtoint ptr addrspace(3) %_arg_acc to i64
; CHECK-NEXT: store i64 [[REG2]], ptr [[REG1]], align 8
; CHECK-NEXT: [[REG3:%[0-9]+]] = ptrtoint ptr %local_args to i64
; CHECK-NEXT: call void @__tsan_cleanup_dynamic_local(i64 [[REG3]], i32 1)
; CHECK-NEXT: call void @__tsan_group_barrier()
  store i32 0, ptr addrspace(3) @WGLocalMem.0, align 4
  store i32 0, ptr addrspace(3) %_arg_acc, align 4
  call void @foo()
  br label %exit

exit: ; preds = %entry
; CHECK: call void @__tsan_cleanup_dynamic_local(i64 [[REG3]], i32 1)
; CHECK-NEXT: call void @__tsan_cleanup_static_local(i64 ptrtoint (ptr addrspace(3) @WGLocalMem.0 to i64), i64 4)
  ret void
}

attributes #0 = { sanitize_thread }
