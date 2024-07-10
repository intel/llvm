; RUN: opt < %s -passes=asan -asan-instrumentation-with-call-threshold=0 -asan-stack=0 -asan-globals=0 -asan-constructor-kind=none -asan-spir-privates=1 -asan-use-after-return=never -S | FileCheck %s

target datalayout = "e-i64:64-v16:16-v24:32-v32:32-v48:64-v96:128-v192:256-v256:256-v512:512-v1024:1024-n8:16:32:64"
target triple = "spir64-unknown-unknown"

%"class.sycl::_V1::range" = type { %"class.sycl::_V1::detail::array" }
%"class.sycl::_V1::detail::array" = type { [1 x i64] }
%"class.sycl::_V1::id" = type { %"class.sycl::_V1::detail::array" }

@__const._ZZZ4mainENKUlRN4sycl3_V17handlerEE_clES2_ENKUlvE_clEv.p = private unnamed_addr addrspace(1) constant [4 x i32] [i32 1, i32 2, i32 3, i32 4], align 4

define spir_func i32 @_Z3fooPii(ptr addrspace(4) %p) {
entry:
  %arrayidx = getelementptr inbounds i32, ptr addrspace(4) %p, i64 0
  %0 = load i32, ptr addrspace(4) %arrayidx, align 4
  ret i32 %0
}

define spir_kernel void @kernel() #0 {
; CHECK-LABEL: define spir_kernel void @kernel
entry:
  %p.i = alloca [4 x i32], align 4
  ; CHECK: %shadow_ptr = call i64 @__asan_mem_to_shadow(i64 %0, i32 0)
  call void @llvm.lifetime.start.p0(i64 16, ptr nonnull %p.i)
  call void @llvm.memcpy.p0.p1.i64(ptr align 4 %p.i, ptr addrspace(1) align 4 @__const._ZZZ4mainENKUlRN4sycl3_V17handlerEE_clES2_ENKUlvE_clEv.p, i64 16, i1 false)
  %arraydecay.i = getelementptr inbounds [4 x i32], ptr %p.i, i64 0, i64 0
  %0 = addrspacecast ptr %arraydecay.i to ptr addrspace(4)
  %call.i = call spir_func i32 @_Z3fooPii(ptr addrspace(4) %0)
  ret void
}

attributes #0 = { mustprogress norecurse nounwind sanitize_address uwtable }
