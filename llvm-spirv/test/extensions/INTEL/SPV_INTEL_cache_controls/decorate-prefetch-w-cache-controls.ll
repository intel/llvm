; RUN: llvm-as %s -o %t.bc
; RUN: llvm-spirv --spirv-ext=+SPV_INTEL_cache_controls -spirv-text %t.bc -o - | FileCheck %s --check-prefix=CHECK-SPIRV

; RUN: llvm-spirv --spirv-ext=+SPV_INTEL_cache_controls %t.bc -o %t.spv
; RUN: llvm-spirv -r %t.spv --spirv-target-env=SPV-IR -o - | llvm-dis -o - | FileCheck %s --check-prefix=CHECK-LLVM

target datalayout = "e-i64:64-v16:16-v24:32-v32:32-v48:64-v96:128-v192:256-v256:256-v512:512-v1024:1024-n8:16:32:64"
target triple = "spir64-unknown-unknown"

$_ZTSZ4mainEUlvE_ = comdat any

; https://github.com/KhronosGroup/SPIRV-Registry/blob/main/extensions/INTEL/SPV_INTEL_cache_controls.asciidoc
; These strings are:
; {CacheControlLoadINTEL_Token:\22CacheLevel,CacheControl\22}
@.str.1 = private unnamed_addr addrspace(1) constant [16 x i8] c"../prefetch.hpp\00", section "llvm.metadata"
@.str.9 = private unnamed_addr addrspace(1) constant [13 x i8] c"{6442:\220,1\22}\00", section "llvm.metadata"
@.str.10 = private unnamed_addr addrspace(1) constant [13 x i8] c"{6442:\221,1\22}\00", section "llvm.metadata"
@.str.11 = private unnamed_addr addrspace(1) constant [13 x i8] c"{6442:\222,3\22}\00", section "llvm.metadata"

; these CHECK-SPIRV check that prefetch's arg is decorated with the appropriate
; CacheLevel and CacheControl values.

; CHECK-SPIRV: Decorate [[PTR_ID1:.*]] CacheControlLoadINTEL 0 1
; CHECK-SPIRV: Decorate [[PTR_ID2:.*]] CacheControlLoadINTEL 1 1
; CHECK-SPIRV: Decorate [[PTR_ID3:.*]] CacheControlLoadINTEL 2 3

; CHECK-SPIRV: ExtInst [[#]] [[#]] [[#]] prefetch [[PTR_ID1]] [[#]]
; CHECK-SPIRV: ExtInst [[#]] [[#]] [[#]] prefetch [[PTR_ID2]] [[#]]
; CHECK-SPIRV: ExtInst [[#]] [[#]] [[#]] prefetch [[PTR_ID3]] [[#]]

; Check that the appropriate !spirv.Decorations are preserved after reverse
; translation

; CHECK-LLVM: %[[CALL1:.*]] = call spir_func ptr addrspace(1) @_Z41__spirv_GenericCastToPtrExplicit_ToGlobal{{.*}} !spirv.Decorations ![[MD1:.*]]
; CHECK-LLVM: call spir_func void @_Z20__spirv_ocl_prefetch{{.*}}(ptr addrspace(1) %[[CALL1]], i64 1)
; CHECK-LLVM: %[[CALL2:.*]] = call spir_func ptr addrspace(1) @_Z41__spirv_GenericCastToPtrExplicit_ToGlobal{{.*}} !spirv.Decorations ![[MD2:.*]]
; CHECK-LLVM: call spir_func void @_Z20__spirv_ocl_prefetch{{.*}}(ptr addrspace(1) %[[CALL2]], i64 1)
; CHECK-LLVM: %[[CALL3:.*]] = call spir_func ptr addrspace(1) @_Z41__spirv_GenericCastToPtrExplicit_ToGlobal{{.*}} !spirv.Decorations ![[MD3:.*]]
; CHECK-LLVM: call spir_func void @_Z20__spirv_ocl_prefetch{{.*}}(ptr addrspace(1) %[[CALL3]], i64 2)


; Function Attrs: convergent norecurse nounwind
define weak_odr dso_local spir_kernel void @_ZTSZ4mainEUlvE_(ptr addrspace(1) noundef align 1 %_arg_dataPtr) local_unnamed_addr comdat !srcloc !5 !kernel_arg_buffer_location !6 !sycl_fixed_targets !7 !sycl_kernel_omit_args !8 {
entry:
  %0 = addrspacecast ptr addrspace(1) %_arg_dataPtr to ptr addrspace(4)
  %call.i.i.i.i = tail call spir_func noundef ptr addrspace(1) @_Z41__spirv_GenericCastToPtrExplicit_ToGlobalPvi(ptr addrspace(4) noundef %0, i32 noundef 5)
  %1 = tail call ptr addrspace(1) @llvm.ptr.annotation.p1.p1(ptr addrspace(1) %call.i.i.i.i, ptr addrspace(1) @.str.9, ptr addrspace(1) @.str.1, i32 76, ptr addrspace(1) null)
  tail call spir_func void @_Z20__spirv_ocl_prefetchPU3AS1Kcm(ptr addrspace(1) noundef %1, i64 noundef 1)
  %arrayidx3.i = getelementptr inbounds i8, ptr addrspace(4) %0, i64 1
  %call.i.i.i13.i = tail call spir_func noundef ptr addrspace(1) @_Z41__spirv_GenericCastToPtrExplicit_ToGlobalPvi(ptr addrspace(4) noundef %arrayidx3.i, i32 noundef 5)
  %2 = tail call ptr addrspace(1) @llvm.ptr.annotation.p1.p1(ptr addrspace(1) %call.i.i.i13.i, ptr addrspace(1) @.str.10, ptr addrspace(1) @.str.1, i32 80, ptr addrspace(1) null)
  tail call spir_func void @_Z20__spirv_ocl_prefetchPU3AS1Kcm(ptr addrspace(1) noundef %2, i64 noundef 1)
  %arrayidx7.i = getelementptr inbounds i8, ptr addrspace(4) %0, i64 2
  %call.i.i.i16.i = tail call spir_func noundef ptr addrspace(1) @_Z41__spirv_GenericCastToPtrExplicit_ToGlobalPvi(ptr addrspace(4) noundef %arrayidx7.i, i32 noundef 5)
  %3 = tail call ptr addrspace(1) @llvm.ptr.annotation.p1.p1(ptr addrspace(1) %call.i.i.i16.i, ptr addrspace(1) @.str.11, ptr addrspace(1) @.str.1, i32 80, ptr addrspace(1) null)
  tail call spir_func void @_Z20__spirv_ocl_prefetchPU3AS1Kcm(ptr addrspace(1) noundef %3, i64 noundef 2)
  ret void
}

; Function Attrs: mustprogress nocallback nofree nosync nounwind willreturn memory(inaccessiblemem: readwrite)
declare ptr addrspace(1) @llvm.ptr.annotation.p1.p1(ptr addrspace(1), ptr addrspace(1), ptr addrspace(1), i32, ptr addrspace(1))

; Function Attrs: convergent nounwind
declare dso_local spir_func void @_Z20__spirv_ocl_prefetchPU3AS1Kcm(ptr addrspace(1) noundef, i64 noundef) local_unnamed_addr

; Function Attrs: convergent mustprogress nofree nounwind willreturn memory(none)
declare dso_local spir_func noundef ptr addrspace(1) @_Z41__spirv_GenericCastToPtrExplicit_ToGlobalPvi(ptr addrspace(4) noundef, i32 noundef) local_unnamed_addr

!llvm.module.flags = !{!0, !1}
!opencl.spir.version = !{!2}
!spirv.Source = !{!3}
!llvm.ident = !{!4}

; CHECK-LLVM: ![[MD1]] = !{![[MD10:.*]]}
; ![[MD10]] = !{i32 6442, i32 0, i32 1}
; CHECK-LLVM: ![[MD2]] = !{![[MD20:.*]]}
; ![[MD20]] = !{i32 6442, i32 1, i32 1}
; CHECK-LLVM: ![[MD3]] = !{![[MD30:.*]]}
; ![[MD30]] = !{i32 6442, i32 2, i32 3}

!0 = !{i32 1, !"wchar_size", i32 4}
!1 = !{i32 7, !"frame-pointer", i32 2}
!2 = !{i32 1, i32 2}
!3 = !{i32 4, i32 100000}
!4 = !{!"clang version 18.0.0"}
!5 = !{i32 1522}
!6 = !{i32 -1}
!7 = !{}
!8 = !{i1 false}
