; RUN: opt -passes=compile-time-properties %s -S | FileCheck %s

target datalayout = "e-i64:64-v16:16-v24:32-v32:32-v48:64-v96:128-v192:256-v256:256-v512:512-v1024:1024-n8:16:32:64"
target triple = "spir64-unknown-unknown"

%struct.foo = type { ptr addrspace(4), ptr addrspace(4), ptr addrspace(4), ptr addrspace(4), ptr addrspace(4), ptr addrspace(4), ptr addrspace(4) }

$_ZTSZ4mainEUlvE_ = comdat any

@.str = private unnamed_addr addrspace(1) constant [16 x i8] c"sycl-properties\00", section "llvm.metadata"
@.str.1 = private unnamed_addr addrspace(1) constant [36 x i8] c"sycl-properties-ptr-annotations.cpp\00", section "llvm.metadata"
@.str.2 = private unnamed_addr addrspace(1) constant [15 x i8] c"sycl-init-mode\00", section "llvm.metadata"
@.str.3 = private unnamed_addr addrspace(1) constant [2 x i8] c"1\00", section "llvm.metadata"
@.str.4 = private unnamed_addr addrspace(1) constant [22 x i8] c"sycl-implement-in-csr\00", section "llvm.metadata"
@.str.5 = private unnamed_addr addrspace(1) constant [5 x i8] c"true\00", section "llvm.metadata"
@.args = private unnamed_addr addrspace(1) constant { ptr addrspace(1), ptr addrspace(1), ptr addrspace(1), ptr addrspace(1) } { ptr addrspace(1) @.str.2, ptr addrspace(1) @.str.3, ptr addrspace(1) @.str.4, ptr addrspace(1) @.str.5 }, section "llvm.metadata"
@.args.6 = private unnamed_addr addrspace(1) constant { ptr addrspace(1), ptr addrspace(1) } { ptr addrspace(1) @.str.2, ptr addrspace(1) @.str.3 }, section "llvm.metadata"
@.args.7 = private unnamed_addr addrspace(1) constant { ptr addrspace(1), ptr addrspace(1) } { ptr addrspace(1) @.str.4, ptr addrspace(1) @.str.5 }, section "llvm.metadata"
@.args.8 = private unnamed_addr addrspace(1) constant { ptr addrspace(1), ptr addrspace(1), ptr addrspace(1), ptr addrspace(1) } { ptr addrspace(1) @.str.2, ptr addrspace(1) @.str.3, ptr addrspace(1) @.str.4, ptr addrspace(1) @.str.5 }, section "llvm.metadata"
@.str.9 = private unnamed_addr addrspace(1) constant [18 x i8] c"sycl-unrecognized\00", section "llvm.metadata"
@.args.10 = private unnamed_addr addrspace(1) constant { ptr addrspace(1), ptr addrspace(1), ptr addrspace(1), ptr addrspace(1) } { ptr addrspace(1) @.str.2, ptr addrspace(1) @.str.3, ptr addrspace(1) @.str.9, ptr addrspace(1) null }, section "llvm.metadata"
@.args.11 = private unnamed_addr addrspace(1) constant { ptr addrspace(1), ptr addrspace(1) } { ptr addrspace(1) @.str.9, ptr addrspace(1) null }, section "llvm.metadata"

;CHECK: @[[NewAnnotStr1:.*]] = private unnamed_addr addrspace(1) constant [24 x i8] c"{6148:\221\22}{6149:\22true\22}\00", section "llvm.metadata"
;CHECK: @[[NewAnnotStr2:.*]] = private unnamed_addr addrspace(1) constant [11 x i8] c"{6148:\221\22}\00", section "llvm.metadata"
;CHECK: @[[NewAnnotStr3:.*]] = private unnamed_addr addrspace(1) constant [14 x i8] c"{6149:\22true\22}\00", section "llvm.metadata"

; Function Attrs: mustprogress norecurse
define weak_odr dso_local spir_kernel void @_ZTSZ4mainEUlvE_() local_unnamed_addr #0 comdat !kernel_arg_buffer_location !7 !sycl_kernel_omit_args !7 {
entry:
  %x.i = alloca %struct.foo, align 8
  %x.ascast.i = addrspacecast ptr %x.i to ptr addrspace(4)
  %0 = addrspacecast ptr %x.i to ptr addrspace(4)
  %1 = call ptr addrspace(4) @llvm.ptr.annotation.p4.p1(ptr addrspace(4) %0, ptr addrspace(1) @.str, ptr addrspace(1) @.str.1, i32 5, ptr addrspace(1) @.args) #2
; CHECK: %{{.*}} = call ptr addrspace(4) @llvm.ptr.annotation.p4.p1(ptr addrspace(4) %0, ptr addrspace(1) @[[NewAnnotStr1]], ptr addrspace(1) @.str.1, i32 5, ptr addrspace(1) null)
  %b.i = getelementptr inbounds %struct.foo, ptr addrspace(4) %x.ascast.i, i64 0, i32 1
  %2 = call ptr addrspace(4) @llvm.ptr.annotation.p4.p1(ptr addrspace(4) %b.i, ptr addrspace(1) @.str, ptr addrspace(1) @.str.1, i32 6, ptr addrspace(1) @.args.6) #2
; CHECK: %{{.*}} = call ptr addrspace(4) @llvm.ptr.annotation.p4.p1(ptr addrspace(4) %b.i, ptr addrspace(1) @[[NewAnnotStr2]], ptr addrspace(1) @.str.1, i32 6, ptr addrspace(1) null)
  %c.i = getelementptr inbounds %struct.foo, ptr addrspace(4) %x.ascast.i, i64 0, i32 2
  %3 = call ptr addrspace(4) @llvm.ptr.annotation.p4.p1(ptr addrspace(4) %c.i, ptr addrspace(1) @.str, ptr addrspace(1) @.str.1, i32 7, ptr addrspace(1) @.args.7) #2
; CHECK: %{{.*}} = call ptr addrspace(4) @llvm.ptr.annotation.p4.p1(ptr addrspace(4) %c.i, ptr addrspace(1) @[[NewAnnotStr3]], ptr addrspace(1) @.str.1, i32 7, ptr addrspace(1) null)
  %d.i = getelementptr inbounds %struct.foo, ptr addrspace(4) %x.ascast.i, i64 0, i32 3
  %4 = call ptr addrspace(4) @llvm.ptr.annotation.p4.p1(ptr addrspace(4) %d.i, ptr addrspace(1) @.str, ptr addrspace(1) @.str.1, i32 8, ptr addrspace(1) @.args.8) #2
; CHECK: %{{.*}} = call ptr addrspace(4) @llvm.ptr.annotation.p4.p1(ptr addrspace(4) %d.i, ptr addrspace(1) @[[NewAnnotStr1]], ptr addrspace(1) @.str.1, i32 8, ptr addrspace(1) null)
  %e.i = getelementptr inbounds %struct.foo, ptr addrspace(4) %x.ascast.i, i64 0, i32 4
  %5 = call ptr addrspace(4) @llvm.ptr.annotation.p4.p1(ptr addrspace(4) %e.i, ptr addrspace(1) @.str, ptr addrspace(1) @.str.1, i32 9, ptr addrspace(1) @.args.10) #2
; CHECK: %{{.*}} = call ptr addrspace(4) @llvm.ptr.annotation.p4.p1(ptr addrspace(4) %e.i, ptr addrspace(1) @[[NewAnnotStr2]], ptr addrspace(1) @.str.1, i32 9, ptr addrspace(1) null)
  %f.i = getelementptr inbounds %struct.foo, ptr addrspace(4) %x.ascast.i, i64 0, i32 5
  %6 = call ptr addrspace(4) @llvm.ptr.annotation.p4.p1(ptr addrspace(4) %f.i, ptr addrspace(1) @.str, ptr addrspace(1) @.str.1, i32 10, ptr addrspace(1) @.args.11) #2
; CHECK-NOT: %{{.*}} = call ptr addrspace(4) @llvm.ptr.annotation.
  %g.i = getelementptr inbounds %struct.foo, ptr addrspace(4) %x.ascast.i, i64 0, i32 6
  %7 = call ptr addrspace(4) @llvm.ptr.annotation.p4.p1(ptr addrspace(4) %g.i, ptr addrspace(1) @.str, ptr addrspace(1) @.str.1, i32 11, ptr addrspace(1) null) #2
  ret void
}

; Function Attrs: inaccessiblememonly nofree nosync nounwind willreturn
declare ptr addrspace(4) @llvm.ptr.annotation.p4.p1(ptr addrspace(4), ptr addrspace(1), ptr addrspace(1), i32, ptr addrspace(1)) #1

attributes #0 = { mustprogress norecurse "frame-pointer"="all" "min-legal-vector-width"="0" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "sycl-module-id"="sycl-properties-ptr-annotations.cpp" "uniform-work-group-size"="true" }
attributes #1 = { inaccessiblememonly nofree nosync nounwind willreturn }
attributes #2 = { nounwind }

!opencl.spir.version = !{!0, !0, !0, !0, !0, !0}
!spirv.Source = !{!1, !1, !1, !1, !1, !1}
!llvm.ident = !{!2, !3, !3, !3, !4, !3}
!llvm.module.flags = !{!5, !6}

!0 = !{i32 1, i32 2}
!1 = !{i32 4, i32 100000}
!2 = !{!"clang version 15.0.0"}
!3 = !{!"clang version 15.0.0"}
!4 = !{!"clang version 15.0.0"}
!5 = !{i32 1, !"wchar_size", i32 4}
!6 = !{i32 7, !"frame-pointer", i32 2}
!7 = !{}
