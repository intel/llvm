; RUN: llvm-as %s -o %t.bc
; RUN: llvm-spirv %t.bc --spirv-ext=+SPV_INTEL_fpga_buffer_location -o %t.spv
; RUN: llvm-spirv %t.spv -to-text -o %t.spt
; RUN: FileCheck < %t.spt %s --check-prefix=CHECK-SPIRV

; RUN: llvm-spirv -r %t.spv -o %t.rev.bc
; RUN: llvm-dis < %t.rev.bc | FileCheck %s --check-prefix=CHECK-LLVM

; CHECK-SPIRV: Decorate [[#Ptr1:]] BufferLocationINTEL 0
; CHECK-SPIRV: Decorate [[#Ptr2:]] BufferLocationINTEL 0

; CHECK-SPIRV: Load [[#]] [[#Ptr1:]]
; CHECK-SPIRV: ReturnValue [[#Ptr1]]

; CHECK-SPIRV: InBoundsPtrAccessChain [[#]] [[#Ptr2]]
; CHECK-SPIRV: Bitcast [[#]] [[#Bitcast:]] [[#Ptr2]]
; CHECK-SPIRV: ReturnValue [[#Bitcast]]

; CHECK-LLVM: %[[#Load:]] = load ptr addrspace(4)
; CHECK-LLVM: %[[#Anno1:]] = call ptr addrspace(4) @llvm.ptr.annotation.p4.p0(ptr addrspace(4) %[[#Load]], ptr @0, ptr undef, i32 undef, ptr undef)
; CHECK-LLVM: ret ptr addrspace(4) %[[#Anno1]]

; CHECK-LLVM: %[[#GEP:]] = getelementptr inbounds %struct.MyIP
; CHECK-LLVM: %[[#Anno2:]] = call ptr addrspace(4) @llvm.ptr.annotation.p4.p0(ptr addrspace(4) %[[#GEP]]
; CHECK-LLVM: %[[#Bitcast:]] = bitcast ptr addrspace(4) %[[#Anno2]] to ptr addrspace(4)
; CHECK-LLVM: ret ptr addrspace(4) %[[#Bitcast]]

; ModuleID = 'test-sycl-spir64-unknown-unknown.bc'
source_filename = "test.cpp"
target datalayout = "e-i64:64-v16:16-v24:32-v32:32-v48:64-v96:128-v192:256-v256:256-v512:512-v1024:1024-n8:16:32:64"
target triple = "spir64-unknown-unknown"

%struct.MyIP = type <{ ptr addrspace(4), i32, [4 x i8] }>

$_ZNK4MyIPclEv = comdat any

$_Z8annotateIiEPT_S1_ = comdat any

$_Z9annotate2IiEPT_S1_ = comdat any

@.str = private unnamed_addr addrspace(1) constant [16 x i8] c"sycl-properties\00", section "llvm.metadata"
@.str.1 = private unnamed_addr addrspace(1) constant [9 x i8] c"test.cpp\00", section "llvm.metadata"
@.str.2 = private unnamed_addr addrspace(1) constant [21 x i8] c"sycl-buffer-location\00", section "llvm.metadata"
@.str.3 = private unnamed_addr addrspace(1) constant [2 x i8] c"0\00", section "llvm.metadata"
@.args = private unnamed_addr addrspace(1) constant { ptr addrspace(1), ptr addrspace(1) } { ptr addrspace(1) @.str.2, ptr addrspace(1) @.str.3 }, section "llvm.metadata"
@.str.4 = private unnamed_addr addrspace(1) constant [11 x i8] c"{5921:\220\22}\00", section "llvm.metadata"

; Function Attrs: convergent mustprogress noinline norecurse nounwind optnone
define linkonce_odr dso_local spir_func void @_ZNK4MyIPclEv(ptr addrspace(4) %this) comdat align 2 !srcloc !8 {
entry:
  %call1 = call spir_func noundef ptr addrspace(4) @_Z8annotateIiEPT_S1_(ptr addrspace(4) noundef %this)
  %call2 = call spir_func noundef ptr addrspace(4) @_Z9annotate2IiEPT_S1_(ptr addrspace(4) noundef %this)
  ret void
}

; Function Attrs: convergent mustprogress noinline norecurse nounwind optnone
define linkonce_odr dso_local spir_func noundef ptr addrspace(4) @_Z8annotateIiEPT_S1_(ptr addrspace(4) noundef %ptr) comdat !srcloc !9 {
entry:
  %retval = alloca ptr addrspace(4), align 8
  %ptr.addr = alloca ptr addrspace(4), align 8
  %retval.ascast = addrspacecast ptr %retval to ptr addrspace(4)
  %ptr.addr.ascast = addrspacecast ptr %ptr.addr to ptr addrspace(4)
  store ptr addrspace(4) %ptr, ptr addrspace(4) %ptr.addr.ascast, align 8
  %0 = load ptr addrspace(4), ptr addrspace(4) %ptr.addr.ascast, align 8
  %1 = call ptr addrspace(4) @llvm.ptr.annotation.p4.p1(ptr addrspace(4) %0, ptr addrspace(1) @.str.4, ptr addrspace(1) @.str.1, i32 25, ptr addrspace(1) null)
  ret ptr addrspace(4) %1
}

; Function Attrs: convergent mustprogress noinline norecurse nounwind optnone
define linkonce_odr dso_local spir_func noundef ptr addrspace(4) @_Z9annotate2IiEPT_S1_(ptr addrspace(4) noundef %ptr) comdat !srcloc !9 {
entry:
  %retval = alloca ptr addrspace(4), align 8
  %ptr.addr = alloca ptr addrspace(4), align 8
  %retval.ascast = addrspacecast ptr %retval to ptr addrspace(4)
  %ptr.addr.ascast = addrspacecast ptr %ptr.addr to ptr addrspace(4)
  store ptr addrspace(4) %ptr, ptr addrspace(4) %ptr.addr.ascast, align 8
  %0 = load ptr addrspace(4), ptr addrspace(4) %ptr.addr.ascast, align 8
  %1 = getelementptr inbounds %struct.MyIP, ptr addrspace(4) %0, i32 0, i32 0
  %2 = call ptr addrspace(4) @llvm.ptr.annotation.p4.p1(ptr addrspace(4) %1, ptr addrspace(1) @.str.4, ptr addrspace(1) @.str.1, i32 25, ptr addrspace(1) null)
  ret ptr addrspace(4) %2
}

; Function Attrs: nocallback nofree nosync nounwind willreturn memory(inaccessiblemem: readwrite)
declare ptr addrspace(4) @llvm.ptr.annotation.p4.p1(ptr addrspace(4), ptr addrspace(1), ptr addrspace(1), i32, ptr addrspace(1))

!llvm.module.flags = !{!0, !1}
!opencl.spir.version = !{!2}
!spirv.Source = !{!3}
!llvm.ident = !{!4}

!0 = !{i32 1, !"wchar_size", i32 4}
!1 = !{i32 7, !"frame-pointer", i32 2}
!2 = !{i32 1, i32 2}
!3 = !{i32 4, i32 100000}
!4 = !{!"Intel(R) oneAPI DPC++/C++ Compiler 2024.2.0 (2024.x.0.YYYYMMDD)"}
!5 = !{i32 717}
!6 = !{i32 -1, i32 -1}
!7 = !{}
!8 = !{i32 1004}
!9 = !{i32 563}
