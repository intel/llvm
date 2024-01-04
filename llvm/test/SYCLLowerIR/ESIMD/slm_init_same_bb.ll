; This test confirms we don't assert for a single slm_init call
; in a basic block with two predecessors.
;
; RUN: opt < %s -passes=LowerESIMD -S | FileCheck %s

target datalayout = "e-i64:64-v16:16-v24:32-v32:32-v48:64-v96:128-v192:256-v256:256-v512:512-v1024:1024-n8:16:32:64"
target triple = "spir64-unknown-unknown"

@__spirv_BuiltInGlobalInvocationId = external dso_local local_unnamed_addr addrspace(1) constant <3 x i64>, align 32

; Function Attrs: convergent nounwind
declare dso_local spir_func void @_Z16__esimd_slm_initj(i32 noundef) local_unnamed_addr

; Function Attrs: nocallback nofree nosync nounwind willreturn memory(inaccessiblemem: readwrite)
declare void @llvm.assume(i1 noundef)

; Function Attrs: nocallback nofree nosync nounwind willreturn memory(argmem: readwrite)
declare void @llvm.lifetime.start.p0(i64 immarg, ptr nocapture) 

; Function Attrs: nocallback nofree nosync nounwind willreturn memory(argmem: readwrite)
declare void @llvm.lifetime.end.p0(i64 immarg, ptr nocapture)

; Function Attrs: convergent norecurse nounwind
define weak_odr dso_local spir_kernel void @foo() local_unnamed_addr #0 !sycl_explicit_simd !0  {
entry:
  %x.i = alloca i32, align 4
  %0 = load i64, ptr addrspace(1) @__spirv_BuiltInGlobalInvocationId, align 32
  %cmp.i.i = icmp ult i64 %0, 2147483648
  tail call void @llvm.assume(i1 %cmp.i.i)
  %tobool.not.i = icmp eq i64 %0, 0
  br i1 %tobool.not.i, label %foo.exit, label %if.then.i

if.then.i:                                        ; preds = %entry
  call void @llvm.lifetime.start.p0(i64 4, ptr nonnull %x.i)
  %1 = addrspacecast ptr %x.i to ptr addrspace(4)
  store volatile i32 0, ptr addrspace(4) %1, align 4
  call void @llvm.lifetime.end.p0(i64 4, ptr nonnull %x.i)
  br label %foo.exit

; CHECK: foo.exit:
; CHECK-NEXT: ret void
foo.exit: ; preds = %entry, %if.then.i
  tail call spir_func void @_Z16__esimd_slm_initj(i32 noundef 100) #4
  ret void
}

attributes #0 = { convergent norecurse nounwind "frame-pointer"="all" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "sycl-module-id"="test.cpp" "sycl-optlevel"="2" "uniform-work-group-size"="true" }

!0 = !{}