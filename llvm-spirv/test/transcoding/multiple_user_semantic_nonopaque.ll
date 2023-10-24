; RUN: llvm-as %s -o %t.bc
; RUN: llvm-spirv %t.bc -spirv-text -o - | FileCheck %s --check-prefix=CHECK-SPIRV

; RUN: llvm-spirv %t.bc -o %t.spv
; RUN: llvm-spirv -r %t.spv -o %t.rev.bc
; RUN: llvm-dis < %t.rev.bc | FileCheck %s --check-prefix=CHECK-LLVM

; Check that even when FPGA memory extensions are enabled - yet we have
; UserSemantic decoration be generated
; RUN: llvm-as %s -o %t.bc
; RUN: llvm-spirv %t.bc --spirv-ext=+SPV_INTEL_fpga_memory_accesses,+SPV_INTEL_fpga_memory_attributes -spirv-text -o - | FileCheck %s --check-prefix=CHECK-SPIRV

; CHECK-SPIRV: Name [[#ClassMember:]] "class.Sample"
; CHECK-SPIRV: Decorate [[#Var:]] UserSemantic "var_annotation_a"
; CHECK-SPIRV: Decorate [[#Var]] UserSemantic "var_annotation_b"
; CHECK-SPIRV: MemberDecorate [[#ClassMember]] 0 UserSemantic "class_annotation_a"
; CHECK-SPIRV: MemberDecorate [[#ClassMember]] 0 UserSemantic "class_annotation_b"
; CHECK-SPIRV: Variable [[#]] [[#Var]] [[#]]

; CHECK-LLVM-DAG: @[[StrA:[0-9_.]+]] = {{.*}}"var_annotation_a\00"
; CHECK-LLVM-DAG: @[[StrB:[0-9_.]+]] = {{.*}}"var_annotation_b\00"
; CHECK-LLVM-DAG: @[[StrStructA:[0-9_.]+]] = {{.*}}"class_annotation_a\00"
; CHECK-LLVM-DAG: @[[StrStructB:[0-9_.]+]] = {{.*}}"class_annotation_b\00"
; CHECK-LLVM: [[#Var:]] = alloca i32, align 4
; CHECK-LLVM: call void @llvm.var.annotation.p0.p0(ptr %[[#Var]], ptr @[[StrA]], ptr undef, i32 undef, ptr undef)
; CHECK-LLVM: call void @llvm.var.annotation.p0.p0(ptr %[[#Var]], ptr @[[StrB]], ptr undef, i32 undef, ptr undef)
; CHECK-LLVM: %[[#StructMember:]] = alloca %class.Sample, align 4
; CHECK-LLVM: %[[#PtrAnn:]] = call ptr @llvm.ptr.annotation.p0.p0(ptr %[[#GEP1:]], ptr @[[StrStructA]], ptr undef, i32 undef, ptr undef)
; CHECK-LLVM: call ptr @llvm.ptr.annotation.p0.p0(ptr %[[#PtrAnn]], ptr @[[StrStructB]], ptr undef, i32 undef, ptr undef)


source_filename = "llvm-link"
target datalayout = "e-i64:64-v16:16-v24:32-v32:32-v48:64-v96:128-v192:256-v256:256-v512:512-v1024:1024-n8:16:32:64"
target triple = "spir64"

%class.Sample = type { i32 }

@.str = private unnamed_addr constant [17 x i8] c"var_annotation_a\00", section "llvm.metadata"
@.str.1 = private unnamed_addr constant [17 x i8] c"/app/example.cpp\00", section "llvm.metadata"
@.str.2 = private unnamed_addr constant [17 x i8] c"var_annotation_b\00", section "llvm.metadata"
@.str.3 = private unnamed_addr constant [19 x i8] c"class_annotation_a\00", section "llvm.metadata"
@.str.4 = private unnamed_addr constant [19 x i8] c"class_annotation_b\00", section "llvm.metadata"

define spir_func void @test() {
  %1 = alloca i32, align 4
  %2 = alloca %class.Sample, align 4
  call void @llvm.var.annotation(ptr %1, ptr @.str, ptr @.str.1, i32 9, ptr null)
  call void @llvm.var.annotation(ptr %1, ptr @.str.2, ptr @.str.1, i32 9, ptr null)
  %3 = load i32, ptr %1, align 4
  call void @_Z3fooi(i32 noundef %3)
  %4 = call ptr @llvm.ptr.annotation.p0(ptr %2, ptr @.str.3, ptr @.str.1, i32 3, ptr null)
  %5 = call ptr @llvm.ptr.annotation.p0(ptr %4, ptr @.str.4, ptr @.str.1, i32 3, ptr null)
  %6 = load i32, ptr %5, align 4
  call void @_Z3fooi(i32 noundef %6)
  ret void
}

declare dso_local void @_Z3fooi(i32 noundef)

declare void @llvm.var.annotation(ptr, ptr, ptr, i32, ptr)

declare ptr @llvm.ptr.annotation.p0(ptr, ptr, ptr, i32, ptr)
