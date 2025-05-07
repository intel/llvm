; RUN: opt < %s -passes=msan -msan-instrumentation-with-call-threshold=0 -msan-eager-checks=1 -msan-spir-privates=1 -msan-poison-stack-with-call=1 -S | FileCheck %s
target datalayout = "e-i64:64-v16:16-v24:32-v32:32-v48:64-v96:128-v192:256-v256:256-v512:512-v1024:1024-n8:16:32:64-G1"
target triple = "spir64-unknown-unknown"

define spir_kernel void @MyKernel() sanitize_memory {
; CHECK-LABEL: define spir_kernel void @MyKernel
entry:
  %array = alloca [4 x i32], align 4
  ; CHECK:      %__private_base = alloca i8, align 1
  ; CHECK-NEXT: call void @__msan_set_private_base(ptr %__private_base)
  ; CHECK-NEXT: %array = alloca [4 x i32], align 4
  ; CHECK-NEXT: call void @__msan_poison_stack(ptr %array, i64 16)
  ret void
}

%"class.sycl::_V1::range" = type { %"class.sycl::_V1::detail::array" }
%"class.sycl::_V1::detail::array" = type { [1 x i64] }

define spir_func void @ByValFunc(ptr noundef byval(%"class.sycl::_V1::range") align 8 %_arg_array12) sanitize_memory {
; CHECK-LABEL: define spir_func void @ByValFunc
entry:
  ; CHECK: %0 = ptrtoint ptr %_arg_array12 to i64
  ; CHECK: %1 = call ptr addrspace(1) @__msan_get_shadow(i64 %0, i32 0, ptr addrspace(2) null)
  ; CHECK: call void @llvm.memset.p1.i64(ptr addrspace(1) align 8 %1, i8 0, i64 8, i1 false)
  %_arg_array12.ascast = addrspacecast ptr %_arg_array12 to ptr addrspace(4)
  ret void
}

define spir_kernel void @ByValKernel(ptr noundef byval(%"class.sycl::_V1::range") align 8 %_arg_array12) sanitize_memory {
; CHECK-LABEL: @ByValKernel
entry:
  ; CHECK:      %__private_base = alloca i8, align 1
  ; CHECK-NEXT: call void @__msan_set_private_base(ptr %__private_base)

  ; CHECK: %_arg_array12.byval = alloca %"class.sycl::_V1::range", align 8
  ; CHECK: call void @__msan_unpoison_stack(ptr %_arg_array12.byval, i64 8), !nosanitize
  ; CHECK: call void @llvm.memcpy.p0.p0.i64(ptr align 8 %_arg_array12.byval, ptr align 8 %_arg_array12, i64 8, i1 false), !nosanitize
  call void @ByValFunc(ptr %_arg_array12)
  ret void
}
