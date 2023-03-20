; RUN: opt -passes=compile-time-properties %s -S | FileCheck %s

target datalayout = "e-i64:64-v16:16-v24:32-v32:32-v48:64-v96:128-v192:256-v256:256-v512:512-v1024:1024-n8:16:32:64"
target triple = "spir64-unknown-unknown"

%struct.foo = type { i32 addrspace(4)*, i32 addrspace(4)*, i32 addrspace(4)*, i32 addrspace(4)*, i32 addrspace(4)*, i32 addrspace(4)*, i32 addrspace(4)* }

$_ZTSZ4mainEUlvE_ = comdat any

@.str = private unnamed_addr addrspace(1) constant [16 x i8] c"sycl-properties\00", section "llvm.metadata"
@.str.1 = private unnamed_addr addrspace(1) constant [19 x i8] c"inc/fpga_utils.hpp\00", section "llvm.metadata"
@.str.2 = private unnamed_addr addrspace(1) constant [23 x i8] c"sycl-latency-anchor-id\00", section "llvm.metadata"
@.str.3 = private unnamed_addr addrspace(1) constant [2 x i8] c"0\00", section "llvm.metadata"
@.args = private unnamed_addr addrspace(1) constant { [23 x i8] addrspace(1)*, [2 x i8] addrspace(1)* } { [23 x i8] addrspace(1)* @.str.2, [2 x i8] addrspace(1)* @.str.3 }, section "llvm.metadata"
@.str.4 = private unnamed_addr addrspace(1) constant [73 x i8] c"{params:5}{cache-size:0}{anchor-id:-1}{target-anchor:0}{type:0}{cycle:0}\00", section "llvm.metadata"
@.str.5 = private unnamed_addr addrspace(1) constant [14 x i8] c"<invalid loc>\00", section "llvm.metadata"
@.str.6 = private unnamed_addr addrspace(1) constant [2 x i8] c"1\00", section "llvm.metadata"
@.str.7 = private unnamed_addr addrspace(1) constant [24 x i8] c"sycl-latency-constraint\00", section "llvm.metadata"
@.str.8 = private unnamed_addr addrspace(1) constant [6 x i8] c"0,1,5\00", section "llvm.metadata"
@.args.9 = private unnamed_addr addrspace(1) constant { [23 x i8] addrspace(1)*, [2 x i8] addrspace(1)*, [24 x i8] addrspace(1)*, [6 x i8] addrspace(1)* } { [23 x i8] addrspace(1)* @.str.2, [2 x i8] addrspace(1)* @.str.6, [24 x i8] addrspace(1)* @.str.7, [6 x i8] addrspace(1)* @.str.8 }, section "llvm.metadata"

;CHECK: @[[NewAnnotStr1:.*]] = private unnamed_addr addrspace(1) constant [11 x i8] c"{6172:\220\22}\00", section "llvm.metadata"
;CHECK: @[[NewAnnotStr2:.*]] = private unnamed_addr addrspace(1) constant [25 x i8] c"{6172:\221\22}{6173:\220,1,5\22}\00", section "llvm.metadata"

; Function Attrs: mustprogress norecurse
define weak_odr dso_local spir_kernel void @_ZTSZ4mainEUlvE_() local_unnamed_addr #0 comdat !kernel_arg_buffer_location !7 !sycl_kernel_omit_args !7 {
entry:
  %x.i = alloca %struct.foo, align 8
  %x.ascast.i = addrspacecast %struct.foo* %x.i to %struct.foo addrspace(4)*
  %0 = bitcast %struct.foo* %x.i to i8*
  %1 = addrspacecast i8* %0 to i8 addrspace(4)*
  %2 = call i8 addrspace(4)* @llvm.ptr.annotation.p4i8.p1i8(i8 addrspace(4)* %1, i8 addrspace(1)* getelementptr inbounds ([16 x i8], [16 x i8] addrspace(1)* @.str, i64 0, i64 0), i8 addrspace(1)* getelementptr inbounds ([19 x i8], [19 x i8] addrspace(1)* @.str.1, i64 0, i64 0), i32 5, i8 addrspace(1)* bitcast ({ [23 x i8] addrspace(1)*, [2 x i8] addrspace(1)* } addrspace(1)* @.args to i8 addrspace(1)*))
; CHECK: %{{.*}} = call i8 addrspace(4)* @llvm.ptr.annotation.p4i8.p1i8(i8 addrspace(4)* %1, i8 addrspace(1)* getelementptr inbounds ([11 x i8], [11 x i8] addrspace(1)* @[[NewAnnotStr1]], i32 0, i32 0), i8 addrspace(1)* getelementptr inbounds ([19 x i8], [19 x i8] addrspace(1)* @.str.1, i64 0, i64 0), i32 5, i8 addrspace(1)* null)
  %3 = bitcast i8 addrspace(4)* %2 to i32 addrspace(4)*
  %4 = load i32, i32 addrspace(4)* %3, align 8
  %b.i = getelementptr inbounds %struct.foo, %struct.foo addrspace(4)* %x.ascast.i, i64 0, i32 1
  %5 = bitcast i32 addrspace(4)* addrspace(4)* %b.i to i8 addrspace(4)*
  %6 = call i8 addrspace(4)* @llvm.ptr.annotation.p4i8.p1i8(i8 addrspace(4)* %5, i8 addrspace(1)* getelementptr inbounds ([16 x i8], [16 x i8] addrspace(1)* @.str, i64 0, i64 0), i8 addrspace(1)* getelementptr inbounds ([19 x i8], [19 x i8] addrspace(1)* @.str.1, i64 0, i64 0), i32 5, i8 addrspace(1)* bitcast ({ [23 x i8] addrspace(1)*, [2 x i8] addrspace(1)*, [24 x i8] addrspace(1)*, [6 x i8] addrspace(1)* } addrspace(1)* @.args.9 to i8 addrspace(1)*))
; CHECK: %{{.*}} = call i8 addrspace(4)* @llvm.ptr.annotation.p4i8.p1i8(i8 addrspace(4)* %5, i8 addrspace(1)* getelementptr inbounds ([25 x i8], [25 x i8] addrspace(1)* @[[NewAnnotStr2]], i32 0, i32 0), i8 addrspace(1)* getelementptr inbounds ([19 x i8], [19 x i8] addrspace(1)* @.str.1, i64 0, i64 0), i32 5, i8 addrspace(1)* null)
  %7 = bitcast i8 addrspace(4)* %6 to i32 addrspace(4)*
  %8 = load i32, i32 addrspace(4)* %7, align 8
  ret void
}

; Function Attrs: inaccessiblememonly nofree nosync nounwind willreturn
declare i8 addrspace(4)* @llvm.ptr.annotation.p4i8.p1i8(i8 addrspace(4)*, i8 addrspace(1)*, i8 addrspace(1)*, i32, i8 addrspace(1)*) #1

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
