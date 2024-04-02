; RUN: opt -passes=esimd-opt-call-conv -S < %s | FileCheck %s
; This test checks the ESIMDOptimizeVecArgCallConvPass optimization with a
; use of the sret argument relying on the address space.

; ModuleID = 'opaque_ptr.bc'
source_filename = "llvm-link"
target datalayout = "e-i64:64-v16:16-v24:32-v32:32-v48:64-v96:128-v192:256-v256:256-v512:512-v1024:1024-n8:16:32:64"
target triple = "spir64-unknown-unknown"

%"class.sycl::_V1::ext::intel::esimd::simd.0" = type { %"class.sycl::_V1::ext::intel::esimd::detail::simd_obj_impl.1" }
%"class.sycl::_V1::ext::intel::esimd::detail::simd_obj_impl.1" = type { <16 x float> }

define linkonce_odr dso_local spir_func void @foo(ptr addrspace(4) noalias sret(%"class.sycl::_V1::ext::intel::esimd::simd.0") align 128 %agg.result,
                                                  ptr noundef byval(%"class.sycl::_V1::ext::intel::esimd::simd.0") align 128 %val) {
; CHECK: [[ALLOCA:%.*]] = alloca <16 x float>, align 64
; CHECK: [[CAST:%.*]] = addrspacecast ptr [[ALLOCA]] to ptr addrspace(4)
; CHECK: call void @llvm.memcpy.p4.p4.i64(ptr addrspace(4) align 128 [[CAST]], ptr addrspace(4) align 128 [[ARGCAST:%.*]], i64 128, i1 false)
; CHECK: [[LOAD:%.*]] = load <16 x float>, ptr [[ALLOCA]], align 64
; CHECK: ret <16 x float> [[LOAD]]

entry:
%val.ascast = addrspacecast ptr %val to ptr addrspace(4)
call void @llvm.memcpy.p4.p4.i64(ptr addrspace(4) align 128 %agg.result, ptr addrspace(4) align 128 %val.ascast, i64 128, i1 false)
ret void
}

; Function Attrs: alwaysinline nocallback nofree nounwind willreturn memory(argmem: readwrite)
declare void @llvm.memcpy.p4.p4.i64(ptr addrspace(4) noalias nocapture writeonly %0, ptr addrspace(4) noalias nocapture readonly %1, i64 %2, i1 immarg %3)