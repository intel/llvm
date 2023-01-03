; RUN: llvm-as -opaque-pointers=0 %s -o %t.bc
; RUN: llvm-spirv %t.bc -opaque-pointers=0 -spirv-text -o - | FileCheck %s --check-prefix=CHECK-SPIRV

; RUN: llvm-spirv %t.bc -opaque-pointers=0 -o %t.spv
; RUN: llvm-spirv -r -emit-opaque-pointers %t.spv -o %t.rev.bc
; Added -opaque-pointers.
; FIXME: Align with the community code when project is ready to enable opaque
; pointers by default
; RUN: llvm-dis -opaque-pointers < %t.rev.bc | FileCheck %s --check-prefix=CHECK-LLVM

; Check that even when FPGA memory extensions are enabled - yet we have
; UserSemantic decoration be generated
; RUN: llvm-as -opaque-pointers=0 %s -o %t.bc
; RUN: llvm-spirv %t.bc -opaque-pointers=0 --spirv-ext=+SPV_INTEL_fpga_memory_accesses,+SPV_INTEL_fpga_memory_attributes -spirv-text -o - | FileCheck %s --check-prefix=CHECK-SPIRV

; CHECK-SPIRV-DAG: Decorate {{[0-9]+}} UserSemantic "42"
; CHECK-SPIRV-DAG: Decorate {{[0-9]+}} UserSemantic "bar"
; CHECK-SPIRV-DAG: Decorate {{[0-9]+}} UserSemantic "{FOO}"
; CHECK-SPIRV-DAG: Decorate {{[0-9]+}} UserSemantic "my_custom_annotations: 30, 60"
; CHECK-SPIRV-DAG: MemberDecorate {{[0-9]+}} 1 UserSemantic "128"
; CHECK-SPIRV-DAG: MemberDecorate {{[0-9]+}} 2 UserSemantic "qux"
; CHECK-SPIRV-DAG: MemberDecorate {{[0-9]+}} 0 UserSemantic "{baz}"
; CHECK-SPIRV-DAG: MemberDecorate {{[0-9]+}} 3 UserSemantic "my_custom_annotations: 20, 60, 80"
; CHECK-SPIRV-DAG: MemberDecorate {{[0-9]+}} 0 UserSemantic "annotation_with_zerointializer: 0, 0, 0"
; CHECK-SPIRV-DAG: MemberDecorate {{[0-9]+}} 0 UserSemantic "annotation_with_false: 0"
; CHECK-SPIRV-DAG: MemberDecorate {{[0-9]+}} 0 UserSemantic "annotation_mixed: 0, 1, 0"
; CHECK-SPIRV-DAG: MemberDecorate {{[0-9]+}} 0 UserSemantic "abc: 1, 2, 3"
; CHECK-SPIRV-DAG: MemberDecorate {{[0-9]+}} 0 UserSemantic "annotation_with_true: 1"

target datalayout = "e-i64:64-v16:16-v24:32-v32:32-v48:64-v96:128-v192:256-v256:256-v512:512-v1024:1024"
target triple = "spir64-unknown-linux"

%class.anon = type { i8 }
%struct.bar = type { i32, i8, float, i8 }
%struct.S = type { i32 }
%struct.S.1 = type { i32 }
%struct.S.2 = type { i32 }
%struct.S.3 = type { i32 }
%struct.MyIP = type { i32 addrspace(4)* }

; CHECK-LLVM-DAG:  [[STR:@[0-9_.]+]] = {{.*}}42
; CHECK-LLVM-DAG: [[STR2:@[0-9_.]+]] = {{.*}}{FOO}
; CHECK-LLVM-DAG: [[STR3:@[0-9_.]+]] = {{.*}}bar
; CHECK-LLVM-DAG: [[STR4:@[0-9_.]+]] = {{.*}}{baz}
; CHECK-LLVM-DAG: [[STR5:@[0-9_.]+]] = {{.*}}128
; CHECK-LLVM-DAG: [[STR6:@[0-9_.]+]] = {{.*}}qux
; CHECK-LLVM-DAG: [[STR7:@[0-9_.]+]] = {{.*}}my_custom_annotations: 30, 60
; CHECK-LLVM-DAG: [[STR8:@[0-9_.]+]] = {{.*}}my_custom_annotations: 20, 60, 80
; CHECK-LLVM-DAG: [[STR9:@[0-9_.]+]] = {{.*}}annotation_with_zerointializer: 0, 0, 0
; CHECK-LLVM-DAG: [[STR10:@[0-9_.]+]] = {{.*}}annotation_with_false: 0
; CHECK-LLVM-DAG: [[STR13:@[0-9_.]+]] = {{.*}}annotation_with_true: 1
; CHECK-LLVM-DAG: [[STR11:@[0-9_.]+]] = {{.*}}"annotation_mixed: 0, 1, 0
; CHECK-LLVM-DAG: [[STR12:@[0-9_.]+]] = {{.*}}"abc: 1, 2, 3
@.str = private unnamed_addr constant [3 x i8] c"42\00", section "llvm.metadata"
@.str.1 = private unnamed_addr constant [23 x i8] c"annotate_attribute.cpp\00", section "llvm.metadata"
@.str.2 = private unnamed_addr constant [6 x i8] c"{FOO}\00", section "llvm.metadata"
@.str.3 = private unnamed_addr constant [4 x i8] c"bar\00", section "llvm.metadata"
@.str.4 = private unnamed_addr constant [6 x i8] c"{baz}\00", section "llvm.metadata"
@.str.5 = private unnamed_addr constant [4 x i8] c"128\00", section "llvm.metadata"
@.str.6 = private unnamed_addr constant [4 x i8] c"qux\00", section "llvm.metadata"
@.str.7 = private unnamed_addr constant [22 x i8] c"my_custom_annotations\00", section "llvm.metadata"
@.str.8 = private unnamed_addr constant [31 x i8] c"annotation_with_zerointializer\00", section "llvm.metadata"
@.str.9 = private unnamed_addr constant [22 x i8] c"annotation_with_false\00", section "llvm.metadata"
@.str.10 = private unnamed_addr constant [17 x i8] c"annotation_mixed\00", section "llvm.metadata"
@.args.0 = private unnamed_addr constant { i32, i32 } { i32 30, i32 60 }, section "llvm.metadata"
@.args.1 = private unnamed_addr constant { i32, i32, i32 } { i32 20, i32 60, i32 80 }, section "llvm.metadata"
@.args.2 = private unnamed_addr constant { i32, i32, i32 } zeroinitializer, section "llvm.metadata"
@.args.3 = private unnamed_addr constant { i1 } zeroinitializer, section "llvm.metadata"
@.args.4 = private unnamed_addr constant { i32, i32, i32 } { i32 0, i32 1, i32 0 }, section "llvm.metadata"
@.args.5 = private unnamed_addr constant { i1 } {i1 true }, section "llvm.metadata"
@.str.11 = private unnamed_addr constant [4 x i8] c"abc\00", section "llvm.metadata"
@.str.12 = private unnamed_addr constant [21 x i8] c"annotation_with_true\00", section "llvm.metadata"
@.str.1.12 = private unnamed_addr constant [9 x i8] c"test.cpp\00", section "llvm.metadata"
@.args = private unnamed_addr constant { i32, i32, i32 } { i32 1, i32 2, i32 3 }, section "llvm.metadata"


; Function Attrs: nounwind
define spir_kernel void @_ZTSZ4mainE15kernel_function() #0 !kernel_arg_addr_space !4 !kernel_arg_access_qual !4 !kernel_arg_type !4 !kernel_arg_base_type !4 !kernel_arg_type_qual !4 {
entry:
  %0 = alloca %class.anon, align 1
  %1 = bitcast %class.anon* %0 to i8*
  call void @llvm.lifetime.start.p0i8(i64 1, i8* %1) #4
  call spir_func void @"_ZZ4mainENK3$_0clEv"(%class.anon* %0)
  %2 = bitcast %class.anon* %0 to i8*
  call void @llvm.lifetime.end.p0i8(i64 1, i8* %2) #4
  ret void
}

; Function Attrs: argmemonly nounwind
declare void @llvm.lifetime.start.p0i8(i64, i8* nocapture) #1

; Function Attrs: inlinehint nounwind
define internal spir_func void @"_ZZ4mainENK3$_0clEv"(%class.anon* %this) #2 align 2 {
entry:
  %this.addr = alloca %class.anon*, align 8
  store %class.anon* %this, %class.anon** %this.addr, align 8, !tbaa !5
  %this1 = load %class.anon*, %class.anon** %this.addr, align 8
  call spir_func void @_Z3foov()
  call spir_func void @_Z3bazv()
  ret void
}

; Function Attrs: argmemonly nounwind
declare void @llvm.lifetime.end.p0i8(i64, i8* nocapture) #1

; Function Attrs: nounwind
define spir_func void @_Z3foov() #3 {
entry:
  %var_one = alloca i32, align 4
  %var_two = alloca i32, align 4
  %var_three = alloca i8, align 1
  %var_four = alloca i8, align 1
  %0 = bitcast i32* %var_one to i8*
  call void @llvm.lifetime.start.p0i8(i64 4, i8* %0) #4
  %var_one1 = bitcast i32* %var_one to i8*
  ; CHECK-LLVM: call void @llvm.var.annotation.p0.p0(ptr %{{.*}}, ptr [[STR]], ptr undef, i32 undef, ptr undef)
  call void @llvm.var.annotation(i8* %var_one1, i8* getelementptr inbounds ([3 x i8], [3 x i8]* @.str, i32 0, i32 0), i8* getelementptr inbounds ([23 x i8], [23 x i8]* @.str.1, i32 0, i32 0), i32 2, i8* undef)
  %1 = bitcast i32* %var_two to i8*
  call void @llvm.lifetime.start.p0i8(i64 4, i8* %1) #4
  %var_two2 = bitcast i32* %var_two to i8*
  ; CHECK-LLVM: call void @llvm.var.annotation.p0.p0(ptr %{{.*}}, ptr [[STR2]], ptr undef, i32 undef, ptr undef)
  call void @llvm.var.annotation(i8* %var_two2, i8* getelementptr inbounds ([6 x i8], [6 x i8]* @.str.2, i32 0, i32 0), i8* getelementptr inbounds ([23 x i8], [23 x i8]* @.str.1, i32 0, i32 0), i32 3, i8* undef)
  call void @llvm.lifetime.start.p0i8(i64 1, i8* %var_three) #4
  ; CHECK-LLVM: call void @llvm.var.annotation.p0.p0(ptr %{{.*}}, ptr [[STR3]], ptr undef, i32 undef, ptr undef)
  call void @llvm.var.annotation(i8* %var_three, i8* getelementptr inbounds ([4 x i8], [4 x i8]* @.str.3, i32 0, i32 0), i8* getelementptr inbounds ([23 x i8], [23 x i8]* @.str.1, i32 0, i32 0), i32 4, i8* undef)
  call void @llvm.lifetime.end.p0i8(i64 1, i8* %var_three) #4
  call void @llvm.lifetime.start.p0i8(i64 1, i8* %var_four) #4
  ; CHECK-LLVM: call void @llvm.var.annotation.p0.p0(ptr %{{.*}}, ptr [[STR7]], ptr undef, i32 undef, ptr undef)
  call void @llvm.var.annotation(i8* %var_four, i8* getelementptr inbounds ([22 x i8], [22 x i8]* @.str.7, i32 0, i32 0), i8* getelementptr inbounds ([23 x i8], [23 x i8]* @.str.1, i32 0, i32 0), i32 4, i8* bitcast ({ i32, i32 }* @.args.0 to i8*))
  call void @llvm.lifetime.end.p0i8(i64 1, i8* %var_four) #4
  %2 = bitcast i32* %var_two to i8*
  call void @llvm.lifetime.end.p0i8(i64 4, i8* %2) #4
  %3 = bitcast i32* %var_one to i8*
  call void @llvm.lifetime.end.p0i8(i64 4, i8* %3) #4
  ret void
}

; Function Attrs: nounwind
declare void @llvm.var.annotation(i8*, i8*, i8*, i32, i8*) #4

; Function Attrs: nounwind
define spir_func void @_Z3bazv() #3 {
entry:
  %s1 = alloca %struct.bar, align 4
  %0 = bitcast %struct.bar* %s1 to i8*
  call void @llvm.lifetime.start.p0i8(i64 8, i8* %0) #4
  ; CHECK-LLVM: %[[FIELD1:.*]] = getelementptr inbounds %struct.bar, ptr %{{[a-zA-Z0-9]+}}, i32 0, i32 0
  ; CHECK-LLVM: call ptr @llvm.ptr.annotation.p0{{.*}}%[[FIELD1]]{{.*}}[[STR4]]
  %f1 = getelementptr inbounds %struct.bar, %struct.bar* %s1, i32 0, i32 0
  %1 = call i32* @llvm.ptr.annotation.p0i32.p0i8(i32* %f1, i8* getelementptr inbounds ([6 x i8], [6 x i8]* @.str.4, i32 0, i32 0), i8* getelementptr inbounds ([23 x i8], [23 x i8]* @.str.1, i32 0, i32 0), i32 8, i8* undef)
  store i32 0, i32* %1, align 4, !tbaa !9
  ; CHECK-LLVM: %[[FIELD2:.*]] = getelementptr inbounds %struct.bar, ptr %{{[a-zA-Z0-9]+}}, i32 0, i32 1
  ; CHECK-LLVM: call ptr @llvm.ptr.annotation.p0{{.*}}%[[FIELD2]]{{.*}}[[STR5]]
  %f2 = getelementptr inbounds %struct.bar, %struct.bar* %s1, i32 0, i32 1
  %2 = call i8* @llvm.ptr.annotation.p0i8.p0i8(i8* %f2, i8* getelementptr inbounds ([4 x i8], [4 x i8]* @.str.5, i32 0, i32 0), i8* getelementptr inbounds ([23 x i8], [23 x i8]* @.str.1, i32 0, i32 0), i32 9, i8* undef)
  store i8 0, i8* %2, align 4, !tbaa !13
  ; CHECK-LLVM: %[[FIELD3:.*]] = getelementptr inbounds %struct.bar, ptr %{{[a-zA-Z0-9]+}}, i32 0, i32 2
  ; CHECK-LLVM: call ptr @llvm.ptr.annotation.p0{{.*}}%[[FIELD3]]{{.*}}[[STR6]]
  %f3 = getelementptr inbounds %struct.bar, %struct.bar* %s1, i32 0, i32 2
  %3 = bitcast float* %f3 to i8*
  %4 = call i8* @llvm.ptr.annotation.p0i8.p0i8(i8* %3, i8* getelementptr inbounds ([4 x i8], [4 x i8]* @.str.6, i32 0, i32 0), i8* getelementptr inbounds ([23 x i8], [23 x i8]* @.str.1, i32 0, i32 0), i32 9, i8* undef)
  %5 = bitcast i8* %4 to float*
  store float 0.000000e+00, float* %5, align 4, !tbaa !14
  ; CHECK-LLVM: %[[FIELD4:.*]] = getelementptr inbounds %struct.bar, ptr %{{[a-zA-Z0-9]+}}, i32 0, i32 3
  ; CHECK-LLVM: call ptr @llvm.ptr.annotation.p0{{.*}}%[[FIELD4]]{{.*}}[[STR8]]
  %f4 = getelementptr inbounds %struct.bar, %struct.bar* %s1, i32 0, i32 3
  %6 = call i8* @llvm.ptr.annotation.p0i8.p0i8(i8* %f4, i8* getelementptr inbounds ([22 x i8], [22 x i8]* @.str.7, i32 0, i32 0), i8* getelementptr inbounds ([23 x i8], [23 x i8]* @.str.1, i32 0, i32 0), i32 9, i8* bitcast ({ i32, i32, i32 }* @.args.1 to i8*))
  store i8 0, i8* %6, align 4, !tbaa !13
  %7 = bitcast %struct.bar* %s1 to i8*
  call void @llvm.lifetime.end.p0i8(i64 12, i8* %7) #4
  ret void
}

; Function Attrs: mustprogress noinline norecurse optnone uwtable
define dso_local noundef i32 @with_zeroinitializer() #0 {
entry:
  %retval = alloca i32, align 4
  %s = alloca %struct.S, align 4
  store i32 0, i32* %retval, align 4
  %a = getelementptr inbounds %struct.S, %struct.S* %s, i32 0, i32 0
  %0 = call i32* @llvm.ptr.annotation.p0i32.p0i8(i32* %a, i8* getelementptr inbounds ([31 x i8], [31 x i8]* @.str.8, i32 0, i32 0), i8* getelementptr inbounds ([23 x i8], [23 x i8]* @.str.1, i32 0, i32 0), i32 3, i8* bitcast ({ i32, i32, i32 }* @.args.2 to i8*))
  ; CHECK-LLVM: %[[FIELD5:.*]] = getelementptr inbounds %struct.S, ptr %{{[a-z]+}}, i32 0, i32 0
  ; CHECK-LLVM: call ptr @llvm.ptr.annotation.p0.p0{{.*}}%[[FIELD5]]{{.*}}[[STR9]]
  %1 = load i32, i32* %0, align 4
  call void @_Z3fooi(i32 noundef %1)
  ret i32 0
}

; Function Attrs: mustprogress noinline norecurse optnone uwtable
define dso_local noundef i32 @with_false() #0 {
entry:
  %retval = alloca i32, align 4
  %s = alloca %struct.S.1, align 4
  store i32 0, i32* %retval, align 4
  %a = getelementptr inbounds %struct.S.1, %struct.S.1* %s, i32 0, i32 0
  %0 = call i32* @llvm.ptr.annotation.p0i32.p0i8(i32* %a, i8* getelementptr inbounds ([22 x i8], [22 x i8]* @.str.9, i32 0, i32 0), i8* getelementptr inbounds ([23 x i8], [23 x i8]* @.str.1, i32 0, i32 0), i32 3, i8* bitcast ({ i1 }* @.args.3 to i8*))
  ; CHECK-LLVM: %[[FIELD6:.*]] = getelementptr inbounds %struct.S.1, ptr %{{[a-z]+}}, i32 0, i32 0
  ; CHECK-LLVM: call ptr @llvm.ptr.annotation.p0{{.*}}%[[FIELD6]]{{.*}}[[STR10]]
  %1 = load i32, i32* %0, align 4
  call void @_Z3fooi(i32 noundef %1)
  ret i32 0
}

; Function Attrs: mustprogress noinline norecurse optnone uwtable
define dso_local noundef i32 @mixed() #0 {
entry:
  %retval = alloca i32, align 4
  %s = alloca %struct.S.2, align 4
  store i32 0, i32* %retval, align 4
  %a = getelementptr inbounds %struct.S.2, %struct.S.2* %s, i32 0, i32 0
  %0 = call i32* @llvm.ptr.annotation.p0i32.p0i8(i32* %a, i8* getelementptr inbounds ([17 x i8], [17 x i8]* @.str.10, i32 0, i32 0), i8* getelementptr inbounds ([23 x i8], [23 x i8]* @.str.1, i32 0, i32 0), i32 3, i8* bitcast ({ i32, i32, i32 }* @.args.4 to i8*))
  ; CHECK-LLVM: %[[FIELD7:.*]] = getelementptr inbounds %struct.S.2, ptr %{{[a-z]+}}, i32 0, i32 0
  ; CHECK-LLVM: call ptr @llvm.ptr.annotation.p0{{.*}}%[[FIELD7]]{{.*}}[[STR11]]
  %1 = load i32, i32* %0, align 4
  call void @_Z3fooi(i32 noundef %1)
  ret i32 0
}

; Function Attrs: mustprogress norecurse
define weak_odr dso_local spir_kernel void @_ZTSZ11TestKernelAvE4MyIP(i32 addrspace(1)* noundef align 4 %0) local_unnamed_addr #5 !kernel_arg_buffer_location !15 !sycl_kernel_omit_args !16 {
  %2 = alloca %struct.MyIP, align 8
  %3 = bitcast %struct.MyIP* %2 to i8*
  call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %3) #4
  %4 = addrspacecast i8* %3 to i8 addrspace(4)*
  %5 = call i8 addrspace(4)* @llvm.ptr.annotation.p4i8.p0i8(i8 addrspace(4)* %4, i8* getelementptr inbounds ([4 x i8], [4 x i8]* @.str.11, i64 0, i64 0), i8* getelementptr inbounds ([9 x i8], [9 x i8]* @.str.1.12, i64 0, i64 0), i32 13, i8* bitcast ({ i32, i32, i32 }* @.args to i8*))
  ; CHECK-LLVM: %[[ALLOCA:.*]] = alloca %struct.MyIP, align 8
  ; CHECK-LLVM: %[[GEP:.*]] = getelementptr inbounds %struct.MyIP, ptr %[[ALLOCA]], i32 0, i32 0
  ; CHECK-LLVM: call ptr @llvm.ptr.annotation.p0.p0(ptr %[[GEP]], ptr [[STR12]], ptr undef, i32 undef, ptr undef)
  %6 = bitcast i8 addrspace(4)* %5 to i32 addrspace(4)* addrspace(4)*
  %7 = addrspacecast i32 addrspace(1)* %0 to i32 addrspace(4)*
  store i32 addrspace(4)* %7, i32 addrspace(4)* addrspace(4)* %6, align 8, !tbaa !17
  %8 = bitcast i8 addrspace(4)* %5 to i32 addrspace(4)* addrspace(4)*
  %9 = load i32 addrspace(4)*, i32 addrspace(4)* addrspace(4)* %8, align 8, !tbaa !17
  %10 = load i32, i32 addrspace(4)* %9, align 4, !tbaa !19
  %11 = shl nsw i32 %10, 1
  store i32 %11, i32 addrspace(4)* %9, align 4, !tbaa !19
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %3) #4
  ret void
}

; Function Attrs: mustprogress noinline norecurse optnone uwtable
define dso_local noundef i32 @with_true() #0 {
entry:
  %retval = alloca i32, align 4
  %s = alloca %struct.S.3, align 4
  store i32 0, i32* %retval, align 4
  %a = getelementptr inbounds %struct.S.3, %struct.S.3* %s, i32 0, i32 0
  %0 = call i32* @llvm.ptr.annotation.p0i32.p0i8(i32* %a, i8* getelementptr inbounds ([21 x i8], [21 x i8]* @.str.12, i32 0, i32 0), i8* getelementptr inbounds ([23 x i8], [23 x i8]* @.str.1, i32 0, i32 0), i32 3, i8* bitcast ({ i1 }* @.args.5 to i8*))
  ; CHECK-LLVM: %[[FIELD8:.*]] = getelementptr inbounds %struct.S.3, ptr %{{[a-z]+}}, i32 0, i32 0
  ; CHECK-LLVM: call ptr @llvm.ptr.annotation.p0{{.*}}%[[FIELD8]]{{.*}}[[STR13]]
  %1 = load i32, i32* %0, align 4
  call void @_Z3fooi(i32 noundef %1)
  ret i32 0
}

declare dso_local void @_Z3fooi(i32 noundef)

; Function Attrs: nounwind
declare i8* @llvm.ptr.annotation.p0i8.p0i8(i8*, i8*, i8*, i32, i8*) #4

; Function Attrs: nounwind
declare i32* @llvm.ptr.annotation.p0i32.p0i8(i32*, i8*, i8*, i32, i8*) #4

; Function Attrs: nounwind
declare i8 addrspace(4)* @llvm.ptr.annotation.p4i8.p0i8(i8 addrspace(4)*, i8*, i8*, i32, i8*) #4

attributes #0 = { nounwind "correctly-rounded-divide-sqrt-fp-math"="false" "disable-tail-calls"="false" "less-precise-fpmad"="false" "min-legal-vector-width"="0" "no-frame-pointer-elim"="false" "no-infs-fp-math"="false" "no-jump-tables"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "no-trapping-math"="false" "stack-protector-buffer-size"="8" "uniform-work-group-size"="true" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #1 = { argmemonly nounwind }
attributes #2 = { inlinehint nounwind "correctly-rounded-divide-sqrt-fp-math"="false" "disable-tail-calls"="false" "less-precise-fpmad"="false" "min-legal-vector-width"="0" "no-frame-pointer-elim"="false" "no-infs-fp-math"="false" "no-jump-tables"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "no-trapping-math"="false" "stack-protector-buffer-size"="8" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #3 = { nounwind optnone noinline "correctly-rounded-divide-sqrt-fp-math"="false" "disable-tail-calls"="false" "less-precise-fpmad"="false" "min-legal-vector-width"="0" "no-frame-pointer-elim"="false" "no-infs-fp-math"="false" "no-jump-tables"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "no-trapping-math"="false" "stack-protector-buffer-size"="8" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #4 = { nounwind }
attributes #5 = { mustprogress norecurse "frame-pointer"="all" "min-legal-vector-width"="0" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "sycl-module-id"="test.cpp" "uniform-work-group-size"="true" }

!llvm.module.flags = !{!0}
!opencl.spir.version = !{!1}
!spirv.Source = !{!2}
!llvm.ident = !{!3}

!0 = !{i32 1, !"wchar_size", i32 4}
!1 = !{i32 1, i32 2}
!2 = !{i32 4, i32 100000}
!3 = !{!"clang version 9.0.0"}
!4 = !{}
!5 = !{!6, !6, i64 0}
!6 = !{!"any pointer", !7, i64 0}
!7 = !{!"omnipotent char", !8, i64 0}
!8 = !{!"Simple C++ TBAA"}
!9 = !{!10, !11, i64 0}
!10 = !{!"_ZTS3bar", !11, i64 0, !7, i64 4, !12, i64 8}
!11 = !{!"int", !7, i64 0}
!12 = !{!"float", !7, i64 0}
!13 = !{!10, !7, i64 4}
!14 = !{!10, !12, i64 8}
!15 = !{i32 -1}
!16 = !{i1 false}
!17 = !{!18, !6, i64 0}
!18 = !{!"_ZTSZ11TestKernelAvE4MyIP", !6, i64 0}
!19 = !{!11, !11, i64 0}
