; RUN: llvm-as %s -o %t.bc
; RUN: llvm-spirv %t.bc --spirv-ext=+SPV_INTEL_fpga_buffer_location -o %t.spv
; RUN: llvm-spirv %t.spv -to-text -o %t.spt
; RUN: FileCheck < %t.spt %s --check-prefix=CHECK-SPIRV

; RUN: llvm-spirv -r %t.spv -o %t.rev.bc
; RUN: llvm-dis < %t.rev.bc | FileCheck %s --check-prefix=CHECK-LLVM

; RUN: llvm-spirv -spirv-text -r %t.spt -o %t.rev.bc
; RUN: llvm-dis < %t.rev.bc | FileCheck %s --check-prefix=CHECK-LLVM

; CHECK-SPIRV: 2 Capability FPGABufferLocationINTEL
; CHECK-SPIRV: 9 Extension "SPV_INTEL_fpga_buffer_location"
; CHECK-SPIRV: 3 Name [[ARGA:[0-9]+]] "a"
; CHECK-SPIRV: 3 Name [[ARGB:[0-9]+]] "b"
; CHECK-SPIRV: 3 Name [[ARGC:[0-9]+]] "c"
; CHECK-SPIRV: 3 Name [[ARGD:[0-9]+]] "d"
; CHECK-SPIRV: 3 Name [[ARGE:[0-9]+]] "e"
; CHECK-SPIRV-NOT: 4 Decorate [[ARGC]] BufferLocationINTEL -1
; CHECK-SPIRV-NOT: 4 Decorate [[ARGC]] BufferLocationINTEL -1
; CHECK-SPIRV: 4 Decorate [[ARGA]] BufferLocationINTEL 1
; CHECK-SPIRV: 4 Decorate [[ARGB]] BufferLocationINTEL 2
; CHECK-SPIRV-NOT: 4 Decorate [[ARGD]] BufferLocationINTEL -1
; CHECK-SPIRV-NOT: 4 Decorate [[ARGE]] BufferLocationINTEL 3
; CHECK-SPIRV-DAG: 4 Decorate {{[0-9]+}} BufferLocationINTEL 123456789

; CHECK-SPIRV: 5 Function
; CHECK-SPIRV: 3 FunctionParameter {{[0-9]+}} [[ARGA]]
; CHECK-SPIRV: 3 FunctionParameter {{[0-9]+}} [[ARGB]]
; CHECK-SPIRV: 3 FunctionParameter {{[0-9]+}} [[ARGC]]
; CHECK-SPIRV: 3 FunctionParameter {{[0-9]+}} [[ARGD]]
; CHECK-SPIRV: 3 FunctionParameter {{[0-9]+}} [[ARGE]]

; ModuleID = 'buffer_location.cl'
target datalayout = "e-i64:64-v16:16-v24:32-v32:32-v48:64-v96:128-v192:256-v256:256-v512:512-v1024:1024"
target triple = "spir64-unknown-unknown"

%struct.MyIP = type { ptr addrspace(4) }

@.str.4 = internal unnamed_addr addrspace(1) constant [19 x i8] c"{5921:\22123456789\22}\00"
@.str.1 = internal unnamed_addr addrspace(1) constant [9 x i8] c"main.cpp\00"
; CHECK-LLVM: @[[ANN_STR:[0-9]+]] = private unnamed_addr constant [33 x i8] c"{sycl-buffer-location:123456789}\00"

; Function Attrs: nounwind
define spir_kernel void @test(ptr addrspace(1) %a, ptr addrspace(1) %b, ptr addrspace(1) %c, i32 %d, i32 %e) local_unnamed_addr !kernel_arg_addr_space !3 !kernel_arg_access_qual !4 !kernel_arg_type !5 !kernel_arg_base_type !5 !kernel_arg_buffer_location !6
; CHECK-LLVM: !kernel_arg_buffer_location ![[BUFLOC_MD:[0-9]+]]
{
entry:
  ret void
}

; test1 : direct on kernel argument
; Function Attrs: norecurse nounwind readnone
define spir_kernel void @test.1(ptr addrspace(4) %a) #0
; CHECK-LLVM: !kernel_arg_buffer_location ![[BUFLOC_MD_TEST1:[0-9]+]]
{
entry:
  %0 = call ptr addrspace(4) @llvm.ptr.annotation.p4.p1(ptr addrspace(4) %a, ptr addrspace(1) getelementptr inbounds ([19 x i8], ptr addrspace(1) @.str.4, i32 0, i32 0), ptr addrspace(1) getelementptr inbounds ([9 x i8], ptr addrspace(1) @.str.1, i32 0, i32 0), i32 7, ptr addrspace(1) null)
  store i8 0, ptr addrspace(4) %0, align 8
  ret void
}

$test.2 = comdat any
; test2 : general
; Function Attrs: convergent mustprogress norecurse
define weak_odr dso_local spir_kernel void @test.2(ptr addrspace(1) align 4 %arg_a) #0 comdat !kernel_arg_buffer_location !7 {
entry:
  %this.addr.i = alloca ptr addrspace(4), align 8
  %arg_a.addr = alloca ptr addrspace(1), align 8
  %MyIP = alloca %struct.MyIP, align 8
  %arg_a.addr.ascast = addrspacecast ptr %arg_a.addr to ptr addrspace(4)
  %MyIP.ascast = addrspacecast ptr %MyIP to ptr addrspace(4)
  store ptr addrspace(1) %arg_a, ptr addrspace(4) %arg_a.addr.ascast, align 8
  %a = getelementptr inbounds %struct.MyIP, ptr addrspace(4) %MyIP.ascast, i32 0, i32 0
  %0 = call ptr addrspace(4) @llvm.ptr.annotation.p4.p1(ptr addrspace(4) %a, ptr addrspace(1) getelementptr inbounds ([33 x i8], ptr addrspace(1) @.str.4, i32 0, i32 0), ptr addrspace(1) getelementptr inbounds ([9 x i8], ptr addrspace(1) @.str.1, i32 0, i32 0), i32 7, ptr addrspace(1) null)
  %b = load ptr addrspace(1), ptr addrspace(4) %arg_a.addr.ascast, align 8
  %1 = addrspacecast ptr addrspace(1) %b to ptr addrspace(4)
  store ptr addrspace(4) %1, ptr addrspace(4) %0, align 8
; CHECK-LLVM: %[[INTRINSIC_CALL:[[:alnum:].]+]] = call ptr addrspace(4) @llvm.ptr.annotation.p4.p0(ptr addrspace(4) %a, ptr @[[ANN_STR]], ptr undef, i32 undef, ptr undef)
; CHECK-LLVM: %[[BITCAST_CALL1:[[:alnum:].]+]] = bitcast ptr addrspace(4) %[[INTRINSIC_CALL]] to ptr addrspace(4)
; CHECK-LLVM: %[[BITCAST_CALL2:[[:alnum:].]+]] = bitcast ptr addrspace(4) %[[BITCAST_CALL1]] to ptr addrspace(4)
; CHECK-LLVM: store ptr addrspace(4) %[[#]], ptr addrspace(4) %[[BITCAST_CALL2]], align 8
  %this.addr.ascast.i = addrspacecast ptr %this.addr.i to ptr addrspace(4)
  store ptr addrspace(4) %MyIP.ascast, ptr addrspace(4) %this.addr.ascast.i, align 8
  %this1.i = load ptr addrspace(4), ptr addrspace(4) %this.addr.ascast.i, align 8
  %a.i = getelementptr inbounds %struct.MyIP, ptr addrspace(4) %this1.i, i32 0, i32 0
  %2 = call ptr addrspace(4) @llvm.ptr.annotation.p4.p1(ptr addrspace(4) %a.i, ptr addrspace(1) getelementptr inbounds ([19 x i8], ptr addrspace(1) @.str.4, i32 0, i32 0), ptr addrspace(1) getelementptr inbounds ([9 x i8], ptr addrspace(1) @.str.1, i32 0, i32 0), i32 7, ptr addrspace(1) null)
  %3 = load ptr addrspace(4), ptr addrspace(4) %2, align 8
; CHECK-LLVM: %[[INTRINSIC_CALL:[[:alnum:].]+]] = call ptr addrspace(4) @llvm.ptr.annotation.p4.p0(ptr addrspace(4) %a.i, ptr @[[ANN_STR]], ptr undef, i32 undef, ptr undef)
; CHECK-LLVM: %[[BITCAST_CALL1:[[:alnum:].]+]] = bitcast ptr addrspace(4) %[[INTRINSIC_CALL]] to ptr addrspace(4)
; CHECK-LLVM: %[[BITCAST_CALL2:[[:alnum:].]+]] = bitcast ptr addrspace(4) %[[BITCAST_CALL1]] to ptr addrspace(4)
; CHECK-LLVM: load ptr addrspace(4), ptr addrspace(4) %[[BITCAST_CALL2]], align 8
  %4 = load i32, ptr addrspace(4) %3, align 4
  %inc.i = add nsw i32 %4, 1
  store i32 %inc.i, ptr addrspace(4) %3, align 4
  ret void
}

; test3 : memcpy
; Function Attrs: convergent mustprogress norecurse
define spir_kernel void @test.3(ptr addrspace(1) align 4 %arg_a, ptr %arg_b) {
entry:
  %this.addr.i = alloca ptr addrspace(4), align 8
  %arg_a.addr = alloca ptr addrspace(1), align 8
  %MyIP = alloca %struct.MyIP, align 8
  %arg_a.addr.ascast = addrspacecast ptr %arg_a.addr to ptr addrspace(4)
  %MyIP.ascast = addrspacecast ptr %MyIP to ptr addrspace(4)
  store ptr addrspace(1) %arg_a, ptr addrspace(4) %arg_a.addr.ascast, align 8
  %a = getelementptr inbounds %struct.MyIP, ptr addrspace(4) %MyIP.ascast, i32 0, i32 0
  %0 = call ptr addrspace(4) @llvm.ptr.annotation.p4.p1(ptr addrspace(4) %a, ptr addrspace(1) getelementptr inbounds ([19 x i8], ptr addrspace(1) @.str.4, i32 0, i32 0), ptr addrspace(1) getelementptr inbounds ([9 x i8], ptr addrspace(1) @.str.1, i32 0, i32 0), i32 7, ptr addrspace(1) null)
  call void @llvm.memcpy.p4.p0(ptr addrspace(4) %0, ptr %arg_b, i64 4, i1 false)
; CHECK-LLVM: %[[INTRINSIC_CALL:[[:alnum:].]+]] = call ptr addrspace(4) @llvm.ptr.annotation.p4.p0(ptr addrspace(4) %a, ptr @[[ANN_STR]], ptr undef, i32 undef, ptr undef)
; CHECK-LLVM: %[[BITCAST:[[:alnum:].]+]] = bitcast ptr addrspace(4) %[[INTRINSIC_CALL]] to ptr addrspace(4)
; CHECK-LLVM: llvm.memcpy.p4.p0.i64(ptr addrspace(4) %[[BITCAST]], ptr %arg_b, i64 4, i1 false)
  ret void
}

; Function Attrs: nocallback nofree nosync nounwind willreturn memory(inaccessiblemem: readwrite)
declare ptr addrspace(4) @llvm.ptr.annotation.p4.p1(ptr addrspace(4), ptr addrspace(1), ptr addrspace(1), i32, ptr addrspace(1)) #1

; Function Attrs: argmemonly nofree nounwind willreturn
declare void @llvm.memcpy.p4.p0(ptr addrspace(4) noalias nocapture writeonly, ptr noalias nocapture readonly, i64, i1 immarg) #2

!opencl.enable.FP_CONTRACT = !{}
!opencl.ocl.version = !{!0}
!opencl.spir.version = !{!0}
!opencl.used.extensions = !{!1}
!opencl.used.optional.core.features = !{!1}
!opencl.compiler.options = !{!1}
!llvm.ident = !{!2}

; CHECK-LLVM: ![[BUFLOC_MD]] = !{i32 1, i32 2, i32 -1, i32 -1, i32 -1}
; CHECK-LLVM: ![[BUFLOC_MD_TEST1]] = !{i32 123456789}
!0 = !{i32 2, i32 0}
!1 = !{}
!2 = !{!""}
!3 = !{i32 1, i32 1, i32 1}
!4 = !{!"none", !"none", !"none"}
!5 = !{!"int*", !"float*", !"int*"}
!6 = !{i32 1, i32 2, i32 -1, i32 -1, i32 3}
!7 = !{i32 -1}
