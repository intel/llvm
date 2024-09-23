; RUN: opt -passes=compile-time-properties -S %s -o %t.ll
; RUN: FileCheck %s -input-file=%t.ll
;
; Tests the translation of "sycl-alignment" to alignment attributes on load/store/non-memory instructions

target triple = "spir64_fpga-unknown-unknown"

%struct.MyIP = type { %class.ann_ptr }
%class.ann_ptr = type { ptr addrspace(4) }

$_ZN7ann_refIiEC2EPi = comdat any
$_ZN7ann_refIiEcvRiEv = comdat any
$_ZN7ann_refIiEC2EPi1= comdat any
$no_load_store = comdat any

@.str = private unnamed_addr addrspace(1) constant [16 x i8] c"sycl-properties\00", section "llvm.metadata"
@.str.1 = private unnamed_addr addrspace(1) constant [9 x i8] c"main.cpp\00", section "llvm.metadata"
@.str.2 = private unnamed_addr addrspace(1) constant [15 x i8] c"sycl-alignment\00", section "llvm.metadata"
@.str.3 = private unnamed_addr addrspace(1) constant [3 x i8] c"64\00", section "llvm.metadata"
@.args = private unnamed_addr addrspace(1) constant { ptr addrspace(1), ptr addrspace(1) } { ptr addrspace(1) @.str.2, ptr addrspace(1) @.str.3 }, section "llvm.metadata"
; CHECK: @[[AnnoStr:.*]] = private unnamed_addr addrspace(1) constant [10 x i8] c"{44:\2264\22}\00"

; Function Attrs: nocallback nofree nosync nounwind willreturn memory(inaccessiblemem: readwrite)
declare ptr addrspace(4) @llvm.ptr.annotation.p4.p1(ptr addrspace(4), ptr addrspace(1), ptr addrspace(1), i32, ptr addrspace(1)) #5

define weak_odr dso_local spir_kernel void @_MyIP(ptr addrspace(1) noundef "sycl-alignment"="64" %_arg_a) {
; CHECK: define{{.*}}@_MyIP{{.*}}align 64{{.*}} {
	ret void
}

; Function Attrs: convergent mustprogress norecurse nounwind
define linkonce_odr dso_local spir_func noundef align 4 dereferenceable(4) ptr addrspace(4) @_ZN7ann_refIiEcvRiEv(ptr addrspace(4) noundef align 8 dereferenceable_or_null(8) %this) #3 comdat align 2 {
entry:
  %retval = alloca ptr addrspace(4), align 8
  %this.addr = alloca ptr addrspace(4), align 8
  %retval.ascast = addrspacecast ptr %retval to ptr addrspace(4)
  %this.addr.ascast = addrspacecast ptr %this.addr to ptr addrspace(4)
  store ptr addrspace(4) %this, ptr addrspace(4) %this.addr.ascast, align 8
  %this1 = load ptr addrspace(4), ptr addrspace(4) %this.addr.ascast, align 8
  %0 = load ptr addrspace(4), ptr addrspace(4) %this1, align 8
  %1 = call ptr addrspace(4) @llvm.ptr.annotation.p4.p1(ptr addrspace(4) %0, ptr addrspace(1) @.str, ptr addrspace(1) @.str.1, i32 22, ptr addrspace(1) @.args)
  %2 = load i32, ptr addrspace(4) %1, align 8
; CHECK: load {{.*}}, align 64
  ret ptr addrspace(4) %1
}

; Function Attrs: convergent norecurse nounwind
define linkonce_odr dso_local spir_func void @_ZN7ann_refIiEC2EPi(ptr addrspace(4) noundef align 8 dereferenceable_or_null(8) %this, ptr addrspace(4) noundef %ptr) unnamed_addr #2 comdat align 2 {
entry:
  %this.addr = alloca ptr addrspace(4), align 8
  %ptr.addr = alloca ptr addrspace(4), align 8
  %this.addr.ascast = addrspacecast ptr %this.addr to ptr addrspace(4)
  %ptr.addr.ascast = addrspacecast ptr %ptr.addr to ptr addrspace(4)
  store ptr addrspace(4) %this, ptr addrspace(4) %this.addr.ascast, align 8
  store ptr addrspace(4) %ptr, ptr addrspace(4) %ptr.addr.ascast, align 8
  %this1 = load ptr addrspace(4), ptr addrspace(4) %this.addr.ascast, align 8
  %0 = load ptr addrspace(4), ptr addrspace(4) %this1, align 8
  %1 = call ptr addrspace(4) @llvm.ptr.annotation.p4.p1(ptr addrspace(4) %0, ptr addrspace(1) @.str, ptr addrspace(1) @.str.1, i32 22, ptr addrspace(1) @.args)
  store i32 5, ptr addrspace(4) %1, align 8
; CHECK: store {{.*}}, align 64
  ret void
}

; Function Attrs: convergent norecurse nounwind
define linkonce_odr dso_local spir_func void @_ZN7ann_refIiEC2EPi1(ptr addrspace(4) noundef align 8 dereferenceable_or_null(8) %this, ptr addrspace(4) noundef %ptr, ptr addrspace(4) %h) comdat align 2 {
entry:
  %this.addr = alloca ptr addrspace(4), align 8
  %ptr.addr = alloca ptr addrspace(4), align 8
  %this.addr.ascast = addrspacecast ptr %this.addr to ptr addrspace(4)
  %ptr.addr.ascast = addrspacecast ptr %ptr.addr to ptr addrspace(4)
  store ptr addrspace(4) %this, ptr addrspace(4) %this.addr.ascast, align 8
  store ptr addrspace(4) %ptr, ptr addrspace(4) %ptr.addr.ascast, align 8
  %this1 = load ptr addrspace(4), ptr addrspace(4) %this.addr.ascast, align 8
  %0 = load ptr addrspace(4), ptr addrspace(4) %this1, align 8
  %1 = call ptr addrspace(4) @llvm.ptr.annotation.p4.p1(ptr addrspace(4) %0, ptr addrspace(1) @.str, ptr addrspace(1) @.str.1, i32 22, ptr addrspace(1) @.args)
  call void @llvm.memcpy.p4.p4.i32(ptr addrspace(4) %1, ptr addrspace(4) %h, i32 1, i1 false)
; CHECK: call void @llvm.memcpy.p4.p4.i32(ptr addrspace(4) align 64 %0, ptr addrspace(4) %h, i32 1, i1 false)
  ret void
}

; Function Attrs: convergent mustprogress noinline norecurse nounwind optnone
define linkonce_odr dso_local spir_func noundef ptr addrspace(4) @no_load_store(ptr addrspace(4) noundef %ptr) comdat align 2 {
entry:
  %retval = alloca ptr addrspace(4), align 8
  %ptr.addr = alloca ptr addrspace(4), align 8
  %retval.ascast = addrspacecast ptr %retval to ptr addrspace(4)
  %ptr.addr.ascast = addrspacecast ptr %ptr.addr to ptr addrspace(4)
  store ptr addrspace(4) %ptr, ptr addrspace(4) %ptr.addr.ascast, align 8
  %0 = load ptr addrspace(4), ptr addrspace(4) %ptr.addr.ascast, align 8
  ; CHECK: %[[AnnoPtr:.*]] = call ptr addrspace(4) @llvm.ptr.annotation.p4.p1(ptr addrspace(4) %0, ptr addrspace(1) @[[AnnoStr]]
  ; CHECK: ret ptr addrspace(4) %[[AnnoPtr]]
  %1 = call ptr addrspace(4) @llvm.ptr.annotation.p4.p1(ptr addrspace(4) %0, ptr addrspace(1) @.str, ptr addrspace(1) @.str.1, i32 73, ptr addrspace(1) @.args)
  ret ptr addrspace(4) %1
}

declare void @llvm.memcpy.p4.p4.i32(ptr addrspace(4), ptr addrspace(4), i32, i1)
