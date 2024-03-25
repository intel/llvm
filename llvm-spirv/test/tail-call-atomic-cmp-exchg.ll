; REQUIRES: pass-plugin
; UNSUPPORTED: target={{.*windows.*}}

; RUN: opt %load_spirv_lib -passes=spirv-to-ocl20 %s -S -o - | FileCheck %s

; Check that tail marker is removed from atomic_compare_exchange_strong_explicit call.

; CHECK: = call spir_func {{.*}}atomic_compare_exchange_strong_explicit

target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-i128:128-f80:128-n8:16:32:64-S128"

%"class.sycl::_V1::id" = type { %"class.sycl::_V1::detail::array" }
%"class.sycl::_V1::detail::array" = type { [1 x i64] }

define spir_kernel void @test(ptr addrspace(1) noundef align 8 %_arg_data_accessor, ptr noundef byval(%"class.sycl::_V1::id") align 8 %_arg_data_accessor4) {
entry:
  %0 = load i64, ptr %_arg_data_accessor4, align 8
  %add.ptr = getelementptr inbounds ptr addrspace(4), ptr addrspace(1) %_arg_data_accessor, i64 %0
  %arrayidx.ascast = addrspacecast ptr addrspace(1) %add.ptr to ptr addrspace(4)
  br label %do.body

do.body:
  %call1 = tail call spir_func noundef i64 @_Z18__spirv_AtomicLoadPKmN5__spv5Scope4FlagENS1_19MemorySemanticsMask4FlagE(ptr addrspace(4) noundef %arrayidx.ascast, i32 noundef 2, i32 noundef 912)
  %1 = inttoptr i64 %call1 to ptr addrspace(4)
  %add.ptr.i = getelementptr inbounds i32, ptr addrspace(4) %1, i64 1
  %2 = ptrtoint ptr addrspace(4) %add.ptr.i to i64
  %call2 = tail call spir_func noundef i64 @_Z29__spirv_AtomicCompareExchangePmN5__spv5Scope4FlagENS0_19MemorySemanticsMask4FlagES4_mm(ptr addrspace(4) noundef %arrayidx.ascast, i32 noundef 2, i32 noundef 912, i32 noundef 912, i64 noundef %2, i64 noundef %call1)
  %3 = icmp eq i64 %call2, %call1
  br i1 %3, label %exit, label %do.body

exit:
  ret void
}

declare spir_func noundef i64 @_Z18__spirv_AtomicLoadPKmN5__spv5Scope4FlagENS1_19MemorySemanticsMask4FlagE(ptr addrspace(4) noundef, i32 noundef, i32 noundef)

declare spir_func noundef i64 @_Z29__spirv_AtomicCompareExchangePmN5__spv5Scope4FlagENS0_19MemorySemanticsMask4FlagES4_mm(ptr addrspace(4) noundef, i32 noundef, i32 noundef, i32 noundef, i64 noundef, i64 noundef)
