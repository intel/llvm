; RUN: opt < %s -passes=asan -asan-instrumentation-with-call-threshold=0 -asan-stack=0 -asan-globals=0 -asan-constructor-kind=none -S | FileCheck %s

target datalayout = "e-i64:64-v16:16-v24:32-v32:32-v48:64-v96:128-v192:256-v256:256-v512:512-v1024:1024-n8:16:32:64"
target triple = "spir64-unknown-unknown"

%"class.sycl::_V1::range" = type { %"class.sycl::_V1::detail::array" }
%"class.sycl::_V1::detail::array" = type { [1 x i64] }
%"class.sycl::_V1::id" = type { %"class.sycl::_V1::detail::array" }

@WGLocalMem = internal addrspace(3) global [64 x i8] poison, align 4

define spir_kernel void @kernel_static_local() sanitize_address {
; CHECK-LABEL: define spir_kernel void @kernel_static_local
entry:
  store i32 0, ptr addrspace(3) @WGLocalMem
  ; CHECK: store ptr addrspace(1) %__asan_launch, ptr addrspace(3) @__AsanLaunchInfo, align 8
  ; CHECK-NEXT: call void @__asan_set_shadow_static_local(i64 ptrtoint (ptr addrspace(3) @WGLocalMem to i64), i64 64, i64 96)
  ; CHECK-NEXT: store i32 0, ptr addrspace(3) @WGLocalMem, align 4
  ; CHECK-NEXT: call void @__asan_unpoison_shadow_static_local(i64 ptrtoint (ptr addrspace(3) @WGLocalMem to i64), i64 64, i64 96)
  ret void
}

define spir_kernel void @kernel_dynamic_local(ptr addrspace(3) noundef align 4 %_arg_acc, ptr noundef byval(%"class.sycl::_V1::range") align 8 %_arg_acc1, ptr noundef byval(%"class.sycl::_V1::range") align 8 %_arg_acc2, ptr noundef byval(%"class.sycl::_V1::id") align 8 %_arg_acc3) sanitize_address {
; CHECK-LABEL: define spir_kernel void @kernel_dynamic_local
entry:
  ; CHECK:      %local_args = alloca i64, align 8
  ; CHECK-NEXT: [[T0:%.*]] = getelementptr i64, ptr %local_args, i32 0
  ; CHECK-NEXT: [[T1:%.*]] = ptrtoint ptr addrspace(3) %_arg_acc to i64
  ; CHECK-NEXT: store i64 [[T1]], ptr [[T0]], align 8
  ; CHECK-NEXT: [[T2:%.*]] = ptrtoint ptr %local_args to i64
  ; CHECK-NEXT: call void @__asan_set_shadow_dynamic_local(i64 [[T2]], i32 1)
  ; CHECK: call void @__asan_unpoison_shadow_dynamic_local(i64 %2, i32 1)
  ret void
}
