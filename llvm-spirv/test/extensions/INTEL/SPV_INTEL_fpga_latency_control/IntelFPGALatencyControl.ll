; RUN: llvm-as %s -o %t.bc
; RUN: llvm-spirv %t.bc --spirv-ext=+SPV_INTEL_fpga_latency_control -o %t.spv
; RUN: llvm-spirv %t.spv -to-text -o %t.spt
; RUN: FileCheck < %t.spt %s --check-prefix=CHECK-SPIRV

; RUN: llvm-spirv -r %t.spv -o %t.rev.bc
; RUN: llvm-dis < %t.rev.bc | FileCheck %s --check-prefix=CHECK-LLVM

; CHECK-SPIRV: Capability FPGALatencyControlINTEL
; CHECK-SPIRV: Extension "SPV_INTEL_fpga_latency_control"
; CHECK-SPIRV: Decorate [[#ARGA:]] LatencyControlLabelINTEL 0
; CHECK-SPIRV: Decorate [[#ARGB:]] LatencyControlLabelINTEL 1
; CHECK-SPIRV: Decorate [[#ARGB]] LatencyControlConstraintINTEL 0 1 5
; CHECK-SPIRV: Bitcast [[#]] [[#OUT1:]] [[#ARGA]]
; CHECK-SPIRV-DAG: Bitcast [[#]] [[#OUT2:]] [[#OUT1]]
; CHECK-SPIRV-DAG: Load [[#]] [[#]] [[#OUT2]] [[#]] [[#]]
; CHECK-SPIRV: Bitcast [[#]] [[#OUT3:]] [[#ARGB]]
; CHECK-SPIRV-DAG: Bitcast [[#]] [[#OUT4:]] [[#OUT3]]
; CHECK-SPIRV-DAG: Load [[#]] [[#]] [[#OUT4]] [[#]] [[#]]

target datalayout = "e-i64:64-v16:16-v24:32-v32:32-v48:64-v96:128-v192:256-v256:256-v512:512-v1024:1024-n8:16:32:64"
target triple = "spir64-unknown-unknown"

%struct.__spirv_Something = type { i32, i32 }

$_ZTSZ4fooEUlvE_ = comdat any

@.str = private unnamed_addr addrspace(1) constant [16 x i8] c"sycl-properties\00", section "llvm.metadata"
@.str.1 = private unnamed_addr addrspace(1) constant [19 x i8] c"inc/fpga_utils.hpp\00", section "llvm.metadata"
@.str.9 = private unnamed_addr addrspace(1) constant [11 x i8] c"{6172:\220\22}\00", section "llvm.metadata"
@.str.10 = private unnamed_addr addrspace(1) constant [25 x i8] c"{6172:\221\22}{6173:\220,1,5\22}\00", section "llvm.metadata"

; CHECK-LLVM: @[[#ANN_STR1:]] = private unnamed_addr constant [27 x i8] c"{sycl-latency-anchor-id:0}\00"
; CHECK-LLVM: @[[#ANN_STR2:]] = private unnamed_addr constant [58 x i8] c"{sycl-latency-anchor-id:1}{sycl-latency-constraint:0,1,5}\00"

; Function Attrs: mustprogress norecurse
define weak_odr dso_local spir_kernel void @_ZTSZ4fooEUlvE_(ptr %0) local_unnamed_addr #0 comdat !kernel_arg_buffer_location !5 !sycl_kernel_omit_args !5 {
entry:
  %1 = alloca ptr, align 8
  store ptr %0, ptr %1, align 8
  %2 = load ptr, ptr %1, align 8
  %3 = getelementptr inbounds %struct.__spirv_Something, ptr %2, i32 0, i32 0
  %4 = bitcast ptr %3 to ptr
; CHECK-LLVM: %[[#ANN_PTR1:]] = getelementptr inbounds %struct.__spirv_Something, ptr %[[#]], i32 0, i32 0
  %5 = call ptr @llvm.ptr.annotation.p0.p1(ptr %4, ptr addrspace(1) @.str.9, ptr addrspace(1) @.str.1, i32 5, ptr addrspace(1) null)
; CHECK-LLVM: call ptr @llvm.ptr.annotation.p0.p0(ptr %[[#ANN_PTR1]], ptr @[[#ANN_STR1]], ptr undef, i32 undef, ptr undef)
  %6 = load i32, ptr %5, align 8
  %7 = load ptr, ptr %1, align 8
  %8 = getelementptr inbounds %struct.__spirv_Something, ptr %7, i32 0, i32 1
  %9 = bitcast ptr %8 to ptr
; CHECK-LLVM: %[[#ANN_PTR2:]] = getelementptr inbounds %struct.__spirv_Something, ptr %[[#]], i32 0, i32 1
  %10 = call ptr @llvm.ptr.annotation.p0.p1(ptr %9, ptr addrspace(1) @.str.10, ptr addrspace(1) @.str.1, i32 5, ptr addrspace(1) null)
; CHECK-LLVM: call ptr @llvm.ptr.annotation.p0.p0(ptr %[[#ANN_PTR2]], ptr @[[#ANN_STR2]], ptr undef, i32 undef, ptr undef)
  %11 = load i32, ptr %10, align 8
  ret void
}

; Function Attrs: nocallback nofree nosync nounwind willreturn memory(inaccessiblemem: readwrite)
declare ptr @llvm.ptr.annotation.p0.p1(ptr, ptr addrspace(1), ptr addrspace(1), i32, ptr addrspace(1)) #1

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
