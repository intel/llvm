; RUN: llvm-as %s -o %t.bc
; RUN: llvm-spirv %t.bc -spirv-text -o - | FileCheck %s --check-prefix=CHECK-SPIRV

; RUN: llvm-spirv %t.bc -o %t.spv
; RUN: llvm-spirv -r %t.spv -o %t.rev.bc
; RUN: llvm-dis -opaque-pointers=0 < %t.rev.bc | FileCheck %s --check-prefix=CHECK-LLVM

; Check that even when FPGA memory extensions are enabled - yet we have
; UserSemantic decoration be generated
; RUN: llvm-as %s -o %t.bc
; RUN: llvm-spirv %t.bc --spirv-ext=+SPV_INTEL_fpga_memory_accesses,+SPV_INTEL_fpga_memory_attributes -spirv-text -o - | FileCheck %s --check-prefix=CHECK-SPIRV

; CHECK-SPIRV-DAG: Name [[#ClassMember:]] "class.Sample"
; CHECK-SPIRV-DAG: Decorate [[#Var:]] UserSemantic "var_annotation_a"
; CHECK-SPIRV-DAG: Decorate [[#Var]] UserSemantic "var_annotation_b"
; CHECK-SPIRV-DAG: MemberDecorate [[#ClassMember]] 0 UserSemantic "class_annotation_a"
; CHECK-SPIRV-DAG: MemberDecorate [[#ClassMember]] 0 UserSemantic "class_annotation_b"
; CHECK-SPIRV: Variable [[#]] [[#Var]] [[#]]

; CHECK-LLVM: @[[StrStructA:[0-9_.]+]] = {{.*}}"class_annotation_a\00"
; CHECK-LLVM: @[[StrStructB:[0-9_.]+]] = {{.*}}"class_annotation_b\00"
; CHECK-LLVM: @[[StrA:[0-9_.]+]] = {{.*}}"var_annotation_a\00"
; CHECK-LLVM: @[[StrB:[0-9_.]+]] = {{.*}}"var_annotation_b\00"
; CHECK-LLVM: %[[#StructMember:]] = alloca %class.Sample, align 4
; CHECK-LLVM: %[[#GEP1:]] = getelementptr inbounds %class.Sample, %class.Sample* %[[#StructMember]], i32 0, i32 0
; CHECK-LLVM: call i32* @llvm.ptr.annotation.p0i32.p0i8(i32* %[[#GEP1:]], i8* getelementptr inbounds ([19 x i8], [19 x i8]* @[[StrStructA]], i32 0, i32 0), i8* undef, i32 undef, i8* undef)
; CHECK-LLVM: %[[#GEP2:]] = getelementptr inbounds %class.Sample, %class.Sample* %[[#StructMember]], i32 0, i32 0
; CHECK-LLVM: call i32* @llvm.ptr.annotation.p0i32.p0i8(i32* %[[#GEP2]], i8* getelementptr inbounds ([19 x i8], [19 x i8]* @[[StrStructB]], i32 0, i32 0), i8* undef, i32 undef, i8* undef)
; CHECK-LLVM: [[#Var:]] = alloca i32, align 4
; CHECK-LLVM: [[#Bitcast1:]] = bitcast i32* %[[#Var]] to i8*
; CHECK-LLVM: call void @llvm.var.annotation.p0i8.p0i8(i8* %[[#Bitcast1]], i8* getelementptr inbounds ([17 x i8], [17 x i8]* @[[StrA]], i32 0, i32 0), i8* undef, i32 undef, i8* undef)
; CHECK-LLVM: [[#Bitcast2:]] = bitcast i32* %[[#Var]] to i8*
; CHECK-LLVM: call void @llvm.var.annotation.p0i8.p0i8(i8* %[[#Bitcast2]], i8* getelementptr inbounds ([17 x i8], [17 x i8]* @[[StrB]], i32 0, i32 0), i8* undef, i32 undef, i8* undef)


source_filename = "llvm-link"
target datalayout = "e-i64:64-v16:16-v24:32-v32:32-v48:64-v96:128-v192:256-v256:256-v512:512-v1024:1024-n8:16:32:64"
target triple = "spir64"

%class.Sample = type { i32 }

@.str = private unnamed_addr constant [19 x i8] c"class_annotation_a\00", section "llvm.metadata"
@.str.1 = private unnamed_addr constant [17 x i8] c"/app/example.cpp\00", section "llvm.metadata"
@.str.2 = private unnamed_addr constant [19 x i8] c"class_annotation_b\00", section "llvm.metadata"
@.str.3 = private unnamed_addr constant [17 x i8] c"var_annotation_a\00", section "llvm.metadata"
@.str.4 = private unnamed_addr constant [17 x i8] c"var_annotation_b\00", section "llvm.metadata"

define spir_func void @test() {
  %1 = alloca %class.Sample, align 4
  %2 = alloca i32, align 4
  %3 = getelementptr inbounds %class.Sample, ptr %1, i32 0, i32 0
  %4 = call ptr @llvm.ptr.annotation.p0(ptr %3, ptr @.str, ptr @.str.1, i32 3, ptr null)
  %5 = call ptr @llvm.ptr.annotation.p0(ptr %4, ptr @.str.2, ptr @.str.1, i32 3, ptr null)
  %6 = load i32, ptr %5, align 4
  call void @_Z3fooi(i32 noundef %6)
  call void @llvm.var.annotation.p0.p0(ptr %2, ptr @.str.3, ptr @.str.1, i32 11, ptr null)
  call void @llvm.var.annotation.p0.p0(ptr %2, ptr @.str.4, ptr @.str.1, i32 11, ptr null)
  store i32 0, ptr %2, align 4
  %7 = load i32, ptr %2, align 4
  call void @_Z3fooi(i32 noundef %7)
  ret void
}

declare dso_local void @_Z3fooi(i32 noundef)

declare ptr @llvm.ptr.annotation.p0(ptr, ptr, ptr, i32, ptr)

declare void @llvm.var.annotation.p0.p0(ptr, ptr, ptr, i32, ptr)
