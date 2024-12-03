; RUN: llvm-as %s -o %t.bc
; RUN: llvm-spirv %t.bc -spirv-text -o - | FileCheck %s --check-prefixes=CHECK-SPIRV,CHECK-SPIRV-TYPED-PTR
; RUN: llvm-spirv %t.bc -spirv-text --spirv-ext=+SPV_KHR_untyped_pointers -o %t.spt
; RUN: FileCheck < %t.spt %s --check-prefixes=CHECK-SPIRV,CHECK-SPIRV-UNTYPED-PTR

; RUN: llvm-spirv %t.bc -o %t.spv
; RUN: spirv-val %t.spv
; RUN: llvm-spirv -r %t.spv -o %t.rev.bc
; RUN: llvm-dis < %t.rev.bc | FileCheck %s --check-prefix=CHECK-LLVM

; RUN: llvm-spirv %t.bc --spirv-ext=+SPV_KHR_untyped_pointers -o %t.spv
; RUN: spirv-val %t.spv
; RUN: llvm-spirv -r %t.spv -o %t.rev.bc
; RUN: llvm-dis < %t.rev.bc | FileCheck %s --check-prefix=CHECK-LLVM

; Check that even when FPGA memory extensions are enabled - yet we have
; UserSemantic decoration be generated
; RUN: llvm-as %s -o %t.bc
; RUN: llvm-spirv %t.bc --spirv-ext=+SPV_INTEL_fpga_memory_accesses,+SPV_INTEL_fpga_memory_attributes -spirv-text -o - | FileCheck %s --check-prefixes=CHECK-SPIRV,CHECK-SPIRV-TYPED-PTR
; RUN: llvm-spirv %t.bc --spirv-ext=+SPV_INTEL_fpga_memory_accesses,+SPV_INTEL_fpga_memory_attributes,+SPV_KHR_untyped_pointers -spirv-text -o - | FileCheck %s --check-prefixes=CHECK-SPIRV,CHECK-SPIRV-UNTYPED-PTR

; CHECK-SPIRV-DAG: Decorate [[#Var:]] UserSemantic "var_annotation_a"
; CHECK-SPIRV-DAG: Decorate [[#Var]] UserSemantic "var_annotation_b2"
; CHECK-SPIRV-DAG: Decorate [[#Gep:]] UserSemantic "class_annotation_a"
; CHECK-SPIRV-DAG: Decorate [[#Gep]] UserSemantic "class_annotation_b2"
; CHECK-SPIRV-TYPED-PTR: Variable [[#]] [[#Var]] [[#]]
; CHECK-SPIRV-TYPED-PTR: InBoundsPtrAccessChain [[#]] [[#Gep]] [[#]]
; CHECK-SPIRV-UNTYPED-PTR: UntypedVariableKHR [[#]] [[#Var]] [[#]]
; CHECK-SPIRV-UNTYPED-PTR: UntypedInBoundsPtrAccessChainKHR [[#]] [[#Gep]] [[#]]


; CHECK-LLVM-DAG: @[[StrStructA:[0-9_.]+]] = {{.*}}"class_annotation_a\00"
; CHECK-LLVM-DAG: @[[StrStructB:[0-9_.]+]] = {{.*}}"class_annotation_b2\00"
; CHECK-LLVM-DAG: @[[StrA:[0-9_.]+]] = {{.*}}"var_annotation_a\00"
; CHECK-LLVM-DAG: @[[StrB:[0-9_.]+]] = {{.*}}"var_annotation_b2\00"
; CHECK-LLVM: %[[#StructMember:]] = alloca %class.Sample, align 4
; CHECK-LLVM: %[[#Var:]] = alloca i32, align 4
; CHECK-LLVM-DAG: %[[#GEP1:]] = getelementptr inbounds %class.Sample, ptr %[[#StructMember]], i32 0, i32 0
; CHECK-LLVM-DAG: %[[#PtrAnn1:]] = call ptr @llvm.ptr.annotation.p0.p0(ptr %[[#GEP1:]], ptr @[[StrStructA]], ptr undef, i32 undef, ptr undef)
; CHECK-LLVM-DAG: %[[#PtrAnn2:]] = call ptr @llvm.ptr.annotation.p0.p0(ptr %[[#PtrAnn1]], ptr @[[StrStructB]], ptr undef, i32 undef, ptr undef)
; CHECK-LLVM-DAG: call void @llvm.var.annotation.p0.p0(ptr %[[#Var]], ptr @[[StrA]], ptr undef, i32 undef, ptr undef)
; CHECK-LLVM-DAG: call void @llvm.var.annotation.p0.p0(ptr %[[#Var]], ptr @[[StrB]], ptr undef, i32 undef, ptr undef)

source_filename = "llvm-link"
target datalayout = "e-i64:64-v16:16-v24:32-v32:32-v48:64-v96:128-v192:256-v256:256-v512:512-v1024:1024-n8:16:32:64"
target triple = "spir64"

%class.Sample = type { i32 }

@.str = private unnamed_addr addrspace(1) constant [19 x i8] c"class_annotation_a\00", section "llvm.metadata"
@.str.1 = private unnamed_addr addrspace(1) constant [17 x i8] c"/app/example.cpp\00", section "llvm.metadata"
@.str.2 = private unnamed_addr addrspace(1) constant [20 x i8] c"class_annotation_b2\00", section "llvm.metadata"
@.str.3 = private unnamed_addr addrspace(1) constant [17 x i8] c"var_annotation_a\00", section "llvm.metadata"
@.str.4 = private unnamed_addr addrspace(1) constant [18 x i8] c"var_annotation_b2\00", section "llvm.metadata"

define spir_func void @test() {
  %1 = alloca %class.Sample, align 4
  %2 = alloca i32, align 4
  %3 = getelementptr inbounds %class.Sample, ptr %1, i32 0, i32 0
  %4 = call ptr @llvm.ptr.annotation.p0.p1(ptr %3, ptr addrspace(1) @.str, ptr addrspace(1) @.str.1, i32 3, ptr addrspace(1) null)
  %5 = call ptr @llvm.ptr.annotation.p0.p1(ptr %4, ptr addrspace(1) @.str.2, ptr addrspace(1) @.str.1, i32 3, ptr addrspace(1) null)
  %6 = load i32, ptr %5, align 4
  call void @_Z3fooi(i32 noundef %6)
  call void @llvm.var.annotation(ptr %2, ptr addrspace(1) @.str.3, ptr addrspace(1) @.str.1, i32 11, ptr addrspace(1) null)
  call void @llvm.var.annotation(ptr %2, ptr addrspace(1) @.str.4, ptr addrspace(1) @.str.1, i32 11, ptr addrspace(1) null)
  store i32 0, ptr %2, align 4
  %7 = load i32, ptr %2, align 4
  call void @_Z3fooi(i32 noundef %7)
  ret void
}

declare dso_local void @_Z3fooi(i32 noundef)

declare ptr @llvm.ptr.annotation.p0.p1(ptr, ptr addrspace(1), ptr addrspace(1), i32, ptr addrspace(1))

declare void @llvm.var.annotation(ptr, ptr addrspace(1), ptr addrspace(1), i32, ptr addrspace(1))
