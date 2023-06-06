; RUN: opt -passes=compile-time-properties %s -S | FileCheck %s

target datalayout = "e-i64:64-v16:16-v24:32-v32:32-v48:64-v96:128-v192:256-v256:256-v512:512-v1024:1024-n8:16:32:64"
target triple = "spir64-unknown-unknown"

%struct.__spirv_Something = type { i32, i32 }

$_ZTSZ4fooEUlvE_ = comdat any

@.str = private unnamed_addr constant [16 x i8] c"sycl-properties\00", section "llvm.metadata"
@.str.1 = private unnamed_addr constant [19 x i8] c"inc/fpga_utils.hpp\00", section "llvm.metadata"
@.str.2 = private unnamed_addr constant [23 x i8] c"sycl-latency-anchor-id\00", section "llvm.metadata"
@.str.3 = private unnamed_addr constant [2 x i8] c"0\00", section "llvm.metadata"
@.args = private unnamed_addr constant { ptr , ptr } { ptr @.str.2, ptr @.str.3 }, section "llvm.metadata"
@.str.6 = private unnamed_addr constant [2 x i8] c"1\00", section "llvm.metadata"
@.str.7 = private unnamed_addr constant [24 x i8] c"sycl-latency-constraint\00", section "llvm.metadata"
@.str.8 = private unnamed_addr constant [6 x i8] c"0,1,5\00", section "llvm.metadata"
@.args.9 = private unnamed_addr constant { ptr , ptr , ptr , ptr } { ptr @.str.2, ptr @.str.6, ptr @.str.7, ptr @.str.8 }, section "llvm.metadata"

;CHECK: @[[NewAnnotStr1:.*]] = private unnamed_addr constant [11 x i8] c"{6172:\220\22}\00"
;CHECK: @[[NewAnnotStr2:.*]] = private unnamed_addr constant [25 x i8] c"{6172:\221\22}{6173:\220,1,5\22}\00"

; Function Attrs: mustprogress norecurse
define weak_odr dso_local spir_kernel void @_ZTSZ4fooEUlvE_(ptr %0) local_unnamed_addr #0 comdat !kernel_arg_buffer_location !5 !sycl_kernel_omit_args !5 {
entry:
  %1 = alloca ptr , align 8
  store ptr %0, ptr %1, align 8
  %2 = load ptr , ptr %1, align 8
  %3 = getelementptr inbounds %struct.__spirv_Something, ptr %2, i32 0, i32 0
  %4 = bitcast ptr %3 to ptr 
  %5 = call ptr @llvm.ptr.annotation.p0.p0(ptr %4, ptr @.str, ptr @.str.1, i32 5, ptr @.args)
; CHECK: %{{.*}} = call ptr @llvm.ptr.annotation.p0.p0(ptr %[[#]], ptr @[[NewAnnotStr1]], ptr @.str.1, i32 5, ptr null)
  %6 = bitcast ptr %5 to ptr 
  %7 = load i32, ptr %6, align 8
  %8 = load ptr , ptr %1, align 8
  %9 = getelementptr inbounds %struct.__spirv_Something, ptr %8, i32 0, i32 1
  %10 = bitcast ptr %9 to ptr 
  %11 = call ptr @llvm.ptr.annotation.p0.p0(ptr %10, ptr @.str, ptr @.str.1, i32 5, ptr @.args.9)
; CHECK: %{{.*}} = call ptr @llvm.ptr.annotation.p0.p0(ptr %[[#]], ptr @[[NewAnnotStr2]], ptr @.str.1, i32 5, ptr null)
  %12 = bitcast ptr %11 to ptr 
  %13 = load i32, ptr %12, align 8
  ret void
}

; Function Attrs: nocallback nofree nosync nounwind willreturn memory(inaccessiblemem: readwrite)
declare ptr @llvm.ptr.annotation.p0.p0(ptr , ptr , ptr , i32, ptr ) #1

attributes #0 = { mustprogress norecurse "frame-pointer"="all" "min-legal-vector-width"="0" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "sycl-module-id"="sycl-properties-ptr-annotations.cpp" "uniform-work-group-size"="true" }
attributes #1 = { nocallback nofree nosync nounwind willreturn memory(inaccessiblemem: readwrite) }

!opencl.spir.version = !{!0, !0, !0, !0, !0, !0}
!spirv.Source = !{!1, !1, !1, !1, !1, !1}
!llvm.ident = !{!2, !2, !2, !2, !2, !2}
!llvm.module.flags = !{!3, !4}

!0 = !{i32 1, i32 2}
!1 = !{i32 4, i32 100000}
!2 = !{!"clang version 15.0.0"}
!3 = !{i32 1, !"wchar_size", i32 4}
!4 = !{i32 7, !"frame-pointer", i32 2}
!5 = !{}
