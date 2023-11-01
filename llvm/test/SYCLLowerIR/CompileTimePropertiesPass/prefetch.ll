; RUN: opt -passes=compile-time-properties %s -S | FileCheck %s

target datalayout = "e-i64:64-v16:16-v24:32-v32:32-v48:64-v96:128-v192:256-v256:256-v512:512-v1024:1024-n8:16:32:64"
target triple = "spir64-unknown-unknown"

$_ZTSZ4mainEUlvE_ = comdat any

@.str = private unnamed_addr addrspace(1) constant [16 x i8] c"sycl-properties\00", section "llvm.metadata"
@.str.1 = private unnamed_addr addrspace(1) constant [16 x i8] c"../prefetch.hpp\00", section "llvm.metadata"
@.str.2 = private unnamed_addr addrspace(1) constant [19 x i8] c"sycl-prefetch-hint\00", section "llvm.metadata"
@.str.3 = private unnamed_addr addrspace(1) constant [2 x i8] c"0\00", section "llvm.metadata"
@.args = private unnamed_addr addrspace(1) constant { ptr addrspace(1), ptr addrspace(1) } { ptr addrspace(1) @.str.2, ptr addrspace(1) @.str.3 }, section "llvm.metadata"
@.str.4 = private unnamed_addr addrspace(1) constant [2 x i8] c"1\00", section "llvm.metadata"
@.args.5 = private unnamed_addr addrspace(1) constant { ptr addrspace(1), ptr addrspace(1) } { ptr addrspace(1) @.str.2, ptr addrspace(1) @.str.4 }, section "llvm.metadata"
@.str.6 = private unnamed_addr addrspace(1) constant [22 x i8] c"sycl-prefetch-hint-nt\00", section "llvm.metadata"
@.str.7 = private unnamed_addr addrspace(1) constant [2 x i8] c"2\00", section "llvm.metadata"
@.args.8 = private unnamed_addr addrspace(1) constant { ptr addrspace(1), ptr addrspace(1) } { ptr addrspace(1) @.str.6, ptr addrspace(1) @.str.7 }, section "llvm.metadata"

; CHECK: @[[NewAnnotStr1:.*]] = private unnamed_addr addrspace(1) constant [13 x i8] c"{6442:\220,1\22}\00"
; CHECK: @[[NewAnnotStr2:.*]] = private unnamed_addr addrspace(1) constant [13 x i8] c"{6442:\221,1\22}\00"
; CHECK: @[[NewAnnotStr3:.*]] = private unnamed_addr addrspace(1) constant [13 x i8] c"{6442:\222,3\22}\00"

; Function Attrs: convergent norecurse nounwind
define weak_odr dso_local spir_kernel void @_ZTSZ4mainEUlvE_(ptr addrspace(1) noundef align 1 %_arg_dataPtr) local_unnamed_addr comdat !srcloc !5 !kernel_arg_buffer_location !6 !sycl_fixed_targets !7 !sycl_kernel_omit_args !8 {
entry:
  %0 = addrspacecast ptr addrspace(1) %_arg_dataPtr to ptr addrspace(4)
  %call.i.i.i.i = tail call spir_func noundef ptr addrspace(1) @_Z41__spirv_GenericCastToPtrExplicit_ToGlobalPvi(ptr addrspace(4) noundef %0, i32 noundef 5)
  %1 = tail call ptr addrspace(1) @llvm.ptr.annotation.p1.p1(ptr addrspace(1) %call.i.i.i.i, ptr addrspace(1) @.str, ptr addrspace(1) @.str.1, i32 76, ptr addrspace(1) @.args)
  ; CHECK  %{{.*}} = call ptr addrspace(1) @llvm.ptr.annotation.p1.p1(ptr {{.*}}, ptr addrspace(1) @[[NewAnnotStr1]]{{.*}}
  tail call spir_func void @_Z20__spirv_ocl_prefetchPU3AS1Kcm(ptr addrspace(1) noundef %1, i64 noundef 1)
  %arrayidx3.i = getelementptr inbounds i8, ptr addrspace(4) %0, i64 1
  %call.i.i.i13.i = tail call spir_func noundef ptr addrspace(1) @_Z41__spirv_GenericCastToPtrExplicit_ToGlobalPvi(ptr addrspace(4) noundef %arrayidx3.i, i32 noundef 5)
  %2 = tail call ptr addrspace(1) @llvm.ptr.annotation.p1.p1(ptr addrspace(1) %call.i.i.i13.i, ptr addrspace(1) @.str, ptr addrspace(1) @.str.1, i32 80, ptr addrspace(1) @.args.5)
  ; CHECK  %{{.*}} = call ptr addrspace(1) @llvm.ptr.annotation.p1.p1(ptr {{.*}}, ptr addrspace(1) @[[NewAnnotStr2]]{{.*}}
  tail call spir_func void @_Z20__spirv_ocl_prefetchPU3AS1Kcm(ptr addrspace(1) noundef %2, i64 noundef 1)
  %arrayidx7.i = getelementptr inbounds i8, ptr addrspace(4) %0, i64 2
  %call.i.i.i16.i = tail call spir_func noundef ptr addrspace(1) @_Z41__spirv_GenericCastToPtrExplicit_ToGlobalPvi(ptr addrspace(4) noundef %arrayidx7.i, i32 noundef 5)
  %3 = tail call ptr addrspace(1) @llvm.ptr.annotation.p1.p1(ptr addrspace(1) %call.i.i.i16.i, ptr addrspace(1) @.str, ptr addrspace(1) @.str.1, i32 80, ptr addrspace(1) @.args.8)
  ; CHECK  %{{.*}} = call ptr addrspace(1) @llvm.ptr.annotation.p1.p1(ptr {{.*}}, ptr addrspace(1) @[[NewAnnotStr3]]{{.*}}
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

!0 = !{i32 1, !"wchar_size", i32 4}
!1 = !{i32 7, !"frame-pointer", i32 2}
!2 = !{i32 1, i32 2}
!3 = !{i32 4, i32 100000}
!4 = !{!"clang version 18.0.0"}
!5 = !{i32 1522}
!6 = !{i32 -1}
!7 = !{}
!8 = !{i1 false}
