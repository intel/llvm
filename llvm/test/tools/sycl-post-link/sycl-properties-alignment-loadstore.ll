; RUN: opt -passes="compile-time-properties" -S %s -o %t.ll
; RUN: FileCheck %s -input-file=%t.ll
;
; Tests the translation of "sycl-alignment" to alignment attributes on load/store

target triple = "spir64_fpga-unknown-unknown"

%struct.MyIP = type { %class.ann_ptr }
%class.ann_ptr = type { i32 addrspace(4)* }

$_ZN7ann_refIiEC2EPi = comdat any
$_ZN7ann_refIiEcvRiEv = comdat any

@.str = private unnamed_addr addrspace(1) constant [16 x i8] c"sycl-properties\00", section "llvm.metadata"
@.str.1 = private unnamed_addr addrspace(1) constant [9 x i8] c"main.cpp\00", section "llvm.metadata"
@.str.2 = private unnamed_addr addrspace(1) constant [15 x i8] c"sycl-alignment\00", section "llvm.metadata"
@.str.3 = private unnamed_addr addrspace(1) constant [3 x i8] c"64\00", section "llvm.metadata"
@.args = private unnamed_addr addrspace(1) constant { [15 x i8] addrspace(1)*, [3 x i8] addrspace(1)* } { [15 x i8] addrspace(1)* @.str.2, [3 x i8] addrspace(1)* @.str.3 }, section "llvm.met
adata"

; Function Attrs: nocallback nofree nosync nounwind willreturn memory(inaccessiblemem: readwrite)
declare i8 addrspace(4)* @llvm.ptr.annotation.p4i8.p1i8(i8 addrspace(4)*, i8 addrspace(1)*, i8 addrspace(1)*, i32, i8 addrspace(1)*) #5

; Function Attrs: convergent mustprogress norecurse nounwind
define linkonce_odr dso_local spir_func noundef align 4 dereferenceable(4) i32 addrspace(4)* @_ZN7ann_refIiEcvRiEv(%class.ann_ptr addrspace(4)* noundef align 8 dereferenceable_or_null(8) %this) #3 comdat align 2 {
entry:
  %retval = alloca i32 addrspace(4)*, align 8
  %this.addr = alloca %class.ann_ptr addrspace(4)*, align 8
  %retval.ascast = addrspacecast i32 addrspace(4)** %retval to i32 addrspace(4)* addrspace(4)*
  %this.addr.ascast = addrspacecast %class.ann_ptr addrspace(4)** %this.addr to %class.ann_ptr addrspace(4)* addrspace(4)*
  store %class.ann_ptr addrspace(4)* %this, %class.ann_ptr addrspace(4)* addrspace(4)* %this.addr.ascast, align 8
  %this1 = load %class.ann_ptr addrspace(4)*, %class.ann_ptr addrspace(4)* addrspace(4)* %this.addr.ascast, align 8
  %p = getelementptr inbounds %class.ann_ptr, %class.ann_ptr addrspace(4)* %this1, i32 0, i32 0
  %0 = bitcast i32 addrspace(4)* addrspace(4)* %p to i8 addrspace(4)*
  %1 = call i8 addrspace(4)* @llvm.ptr.annotation.p4i8.p1i8(i8 addrspace(4)* %0, i8 addrspace(1)* getelementptr inbounds ([16 x i8], [16 x i8] addrspace(1)* @.str, i32 0, i32 0), i8 addrspace(1)* getelementptr inbounds ([9 x i8], [9 x i8] addrspace(1)* @.str.1, i32 0, i32 0), i32 22, i8 addrspace(1)* bitcast ({ [15 x i8] addrspace(1)*, [3 x i8] addrspace(1)* } addrspace(1)* @.args to i8 addrspace(1)*))
  %2 = bitcast i8 addrspace(4)* %1 to i32 addrspace(4)* addrspace(4)*
  %3 = load i32 addrspace(4)*, i32 addrspace(4)* addrspace(4)* %2, align 8
; CHECK: load i32 addrspace(4)*, i32 addrspace(4)* addrspace(4)* %2, align 64
  ret i32 addrspace(4)* %3
}

; Function Attrs: convergent norecurse nounwind
define linkonce_odr dso_local spir_func void @_ZN7ann_refIiEC2EPi(%class.ann_ptr addrspace(4)* noundef align 8 dereferenceable_or_null(8) %this, i32 addrspace(4)* noundef %ptr) unnamed_addr #2 comdat align 2 {
entry:
  %this.addr = alloca %class.ann_ptr addrspace(4)*, align 8
  %ptr.addr = alloca i32 addrspace(4)*, align 8
  %this.addr.ascast = addrspacecast %class.ann_ptr addrspace(4)** %this.addr to %class.ann_ptr addrspace(4)* addrspace(4)*
  %ptr.addr.ascast = addrspacecast i32 addrspace(4)** %ptr.addr to i32 addrspace(4)* addrspace(4)*
  store %class.ann_ptr addrspace(4)* %this, %class.ann_ptr addrspace(4)* addrspace(4)* %this.addr.ascast, align 8
  store i32 addrspace(4)* %ptr, i32 addrspace(4)* addrspace(4)* %ptr.addr.ascast, align 8
  %this1 = load %class.ann_ptr addrspace(4)*, %class.ann_ptr addrspace(4)* addrspace(4)* %this.addr.ascast, align 8
  %p = getelementptr inbounds %class.ann_ptr, %class.ann_ptr addrspace(4)* %this1, i32 0, i32 0
  %0 = bitcast i32 addrspace(4)* addrspace(4)* %p to i8 addrspace(4)*
  %1 = call i8 addrspace(4)* @llvm.ptr.annotation.p4i8.p1i8(i8 addrspace(4)* %0, i8 addrspace(1)* getelementptr inbounds ([16 x i8], [16 x i8] addrspace(1)* @.str, i32 0, i32 0), i8 addrspace(1)* getelementptr inbounds ([9 x i8], [9 x i8] addrspace(1)* @.str.1, i32 0, i32 0), i32 22, i8 addrspace(1)* bitcast ({ [15 x i8] addrspace(1)*, [3 x i8] addrspace(1)* } addrspace(1)* @.args to i8 addrspace(1)*))
  %2 = bitcast i8 addrspace(4)* %1 to i32 addrspace(4)* addrspace(4)*
  %3 = load i32 addrspace(4)*, i32 addrspace(4)* addrspace(4)* %ptr.addr.ascast, align 8
  store i32 addrspace(4)* %3, i32 addrspace(4)* addrspace(4)* %2, align 8
; CHECK: store i32 addrspace(4)* %3, i32 addrspace(4)* addrspace(4)* %2, align 64
  ret void
}
