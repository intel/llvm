; RUN: llvm-as  %s -o %t.bc
; RUN: llvm-spirv %t.bc -spirv-text -o - | FileCheck %s --check-prefix=CHECK-SPIRV

; RUN: llvm-spirv %t.bc -o %t.spv
; RUN: llvm-spirv -r %t.spv -o %t.rev.bc
; RUN: llvm-dis < %t.rev.bc | FileCheck %s --check-prefix=CHECK-LLVM

; Check that even when FPGA memory extensions are enabled - yet we have
; UserSemantic decoration be generated
; RUN: llvm-as %s -o %t.bc
; RUN: llvm-spirv %t.bc --spirv-ext=+SPV_INTEL_fpga_memory_accesses,+SPV_INTEL_fpga_memory_attributes -spirv-text -o - | FileCheck %s --check-prefix=CHECK-SPIRV

; Check SPIR-V versions in a format magic number + version
; CHECK-SPIRV: 119734787 66560

; CHECK-SPIRV-DAG: Decorate {{[0-9]+}} UserSemantic "42"
; CHECK-SPIRV-DAG: Decorate {{[0-9]+}} UserSemantic "bar"
; CHECK-SPIRV-DAG: Decorate {{[0-9]+}} UserSemantic "{FOO}"
; CHECK-SPIRV-DAG: Decorate {{[0-9]+}} UserSemantic "my_custom_annotations: 30, 60"
; CHECK-SPIRV-DAG: Decorate {{[0-9]+}} UserSemantic "128"
; CHECK-SPIRV-DAG: Decorate {{[0-9]+}} UserSemantic "qux"
; CHECK-SPIRV-DAG: Decorate {{[0-9]+}} UserSemantic "{baz}"
; CHECK-SPIRV-DAG: Decorate {{[0-9]+}} UserSemantic "my_custom_annotations: 20, 60, 80"
; CHECK-SPIRV-DAG: Decorate {{[0-9]+}} UserSemantic "annotation_with_zerointializer: 0, 0, 0"
; CHECK-SPIRV-DAG: Decorate {{[0-9]+}} UserSemantic "annotation_with_false: 0"
; CHECK-SPIRV-DAG: Decorate {{[0-9]+}} UserSemantic "annotation_mixed: 0, 1, 0"
; CHECK-SPIRV-DAG: Decorate {{[0-9]+}} UserSemantic "abc: 1, 2, 3"
; CHECK-SPIRV-DAG: Decorate {{[0-9]+}} UserSemantic "annotation_with_true: 1"

target datalayout = "e-i64:64-v16:16-v24:32-v32:32-v48:64-v96:128-v192:256-v256:256-v512:512-v1024:1024"
target triple = "spir64-unknown-linux"

%class.anon = type { i8 }
%struct.bar = type { i32, i8, float, i8 }
%struct.S = type { i32 }
%struct.S.1 = type { i32 }
%struct.S.2 = type { i32 }
%struct.S.3 = type { i32 }
%struct.MyIP = type { ptr addrspace(4) }

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
@.str = private unnamed_addr addrspace(1) constant [3 x i8] c"42\00", section "llvm.metadata"
@.str.1 = private unnamed_addr addrspace(1) constant [23 x i8] c"annotate_attribute.cpp\00", section "llvm.metadata"
@.str.2 = private unnamed_addr addrspace(1) constant [6 x i8] c"{FOO}\00", section "llvm.metadata"
@.str.3 = private unnamed_addr addrspace(1) constant [4 x i8] c"bar\00", section "llvm.metadata"
@.str.4 = private unnamed_addr addrspace(1) constant [6 x i8] c"{baz}\00", section "llvm.metadata"
@.str.5 = private unnamed_addr addrspace(1) constant [4 x i8] c"128\00", section "llvm.metadata"
@.str.6 = private unnamed_addr addrspace(1) constant [4 x i8] c"qux\00", section "llvm.metadata"
@.str.7 = private unnamed_addr addrspace(1) constant [22 x i8] c"my_custom_annotations\00", section "llvm.metadata"
@.str.8 = private unnamed_addr addrspace(1) constant [31 x i8] c"annotation_with_zerointializer\00", section "llvm.metadata"
@.str.9 = private unnamed_addr addrspace(1) constant [22 x i8] c"annotation_with_false\00", section "llvm.metadata"
@.str.10 = private unnamed_addr addrspace(1) constant [17 x i8] c"annotation_mixed\00", section "llvm.metadata"
@.args.0 = private unnamed_addr addrspace(1) constant { i32, i32 } { i32 30, i32 60 }, section "llvm.metadata"
@.args.1 = private unnamed_addr addrspace(1) constant { i32, i32, i32 } { i32 20, i32 60, i32 80 }, section "llvm.metadata"
@.args.2 = private unnamed_addr addrspace(1) constant { i32, i32, i32 } zeroinitializer, section "llvm.metadata"
@.args.3 = private unnamed_addr addrspace(1) constant { i1 } zeroinitializer, section "llvm.metadata"
@.args.4 = private unnamed_addr addrspace(1) constant { i32, i32, i32 } { i32 0, i32 1, i32 0 }, section "llvm.metadata"
@.args.5 = private unnamed_addr addrspace(1) constant { i1 } {i1 true }, section "llvm.metadata"
@.str.11 = private unnamed_addr addrspace(1) constant [4 x i8] c"abc\00", section "llvm.metadata"
@.str.12 = private unnamed_addr addrspace(1) constant [21 x i8] c"annotation_with_true\00", section "llvm.metadata"
@.str.1.12 = private unnamed_addr addrspace(1) constant [9 x i8] c"test.cpp\00", section "llvm.metadata"
@.args = private unnamed_addr addrspace(1) constant { i32, i32, i32 } { i32 1, i32 2, i32 3 }, section "llvm.metadata"


; Function Attrs: nounwind
define spir_kernel void @_ZTSZ4mainE15kernel_function() #0 !kernel_arg_addr_space !4 !kernel_arg_access_qual !4 !kernel_arg_type !4 !kernel_arg_base_type !4 !kernel_arg_type_qual !4 {
entry:
  %0 = alloca %class.anon, align 1
  call void @llvm.lifetime.start.p0(i64 1, ptr %0) #4
  call spir_func void @"_ZZ4mainENK3$_0clEv"(ptr %0)
  call void @llvm.lifetime.end.p0(i64 1, ptr %0) #4
  ret void
}

; Function Attrs: argmemonly nounwind
declare void @llvm.lifetime.start.p0(i64, ptr nocapture) #1

; Function Attrs: inlinehint nounwind
define internal spir_func void @"_ZZ4mainENK3$_0clEv"(ptr %this) #2 align 2 {
entry:
  %this.addr = alloca ptr, align 8
  store ptr %this, ptr %this.addr, align 8, !tbaa !5
  %this1 = load ptr, ptr %this.addr, align 8
  call spir_func void @_Z3foov()
  call spir_func void @_Z3bazv()
  ret void
}

; Function Attrs: argmemonly nounwind
declare void @llvm.lifetime.end.p0(i64, ptr nocapture) #1

; Function Attrs: nounwind
define spir_func void @_Z3foov() #3 {
entry:
  %var_one = alloca i32, align 4
  %var_two = alloca i32, align 4
  %var_three = alloca i8, align 1
  %var_four = alloca i8, align 1
  call void @llvm.lifetime.start.p0(i64 4, ptr %var_one) #4
  ; CHECK-LLVM: call void @llvm.var.annotation.p0.p0(ptr %{{.*}}, ptr [[STR]], ptr undef, i32 undef, ptr undef)
  call void @llvm.var.annotation(ptr %var_one, ptr addrspace(1) @.str, ptr addrspace(1) @.str.1, i32 2, ptr addrspace(1) undef)
  call void @llvm.lifetime.start.p0(i64 4, ptr %var_two) #4
  ; CHECK-LLVM: call void @llvm.var.annotation.p0.p0(ptr %{{.*}}, ptr [[STR2]], ptr undef, i32 undef, ptr undef)
  call void @llvm.var.annotation(ptr %var_two, ptr addrspace(1) @.str.2, ptr addrspace(1) @.str.1, i32 3, ptr addrspace(1) undef)
  call void @llvm.lifetime.start.p0(i64 1, ptr %var_three) #4
  ; CHECK-LLVM: call void @llvm.var.annotation.p0.p0(ptr %{{.*}}, ptr [[STR3]], ptr undef, i32 undef, ptr undef)
  call void @llvm.var.annotation(ptr %var_three, ptr addrspace(1) @.str.3, ptr addrspace(1) @.str.1, i32 4, ptr addrspace(1) undef)
  call void @llvm.lifetime.end.p0(i64 1, ptr %var_three) #4
  call void @llvm.lifetime.start.p0(i64 1, ptr %var_four) #4
  ; CHECK-LLVM: call void @llvm.var.annotation.p0.p0(ptr %{{.*}}, ptr [[STR7]], ptr undef, i32 undef, ptr undef)
  call void @llvm.var.annotation(ptr %var_four, ptr addrspace(1) @.str.7, ptr addrspace(1) @.str.1, i32 4, ptr addrspace(1) @.args.0)
  call void @llvm.lifetime.end.p0(i64 1, ptr %var_four) #4
  call void @llvm.lifetime.end.p0(i64 4, ptr %var_two) #4
  call void @llvm.lifetime.end.p0(i64 4, ptr %var_one) #4
  ret void
}

; Function Attrs: nounwind
declare void @llvm.var.annotation(ptr, ptr addrspace(1), ptr addrspace(1), i32, ptr addrspace(1)) #4

; Function Attrs: nounwind
define spir_func void @_Z3bazv() #3 {
entry:
  %s1 = alloca %struct.bar, align 4
  call void @llvm.lifetime.start.p0(i64 8, ptr %s1) #4
  ; CHECK-LLVM: %[[FIELD1:.*]] = getelementptr inbounds %struct.bar, ptr %{{[a-zA-Z0-9]+}}, i32 0, i32 0
  ; CHECK-LLVM: call ptr @llvm.ptr.annotation.p0{{.*}}%[[FIELD1]]{{.*}}[[STR4]]
  %f1 = getelementptr inbounds %struct.bar, ptr %s1, i32 0, i32 0
  %0 = call ptr @llvm.ptr.annotation.p0.p1(ptr %f1, ptr addrspace(1) @.str.4, ptr addrspace(1) @.str.1, i32 8, ptr addrspace(1) undef)
  store i32 0, ptr %0, align 4, !tbaa !9
  ; CHECK-LLVM: %[[FIELD2:.*]] = getelementptr inbounds %struct.bar, ptr %{{[a-zA-Z0-9]+}}, i32 0, i32 1
  ; CHECK-LLVM: call ptr @llvm.ptr.annotation.p0{{.*}}%[[FIELD2]]{{.*}}[[STR5]]
  %f2 = getelementptr inbounds %struct.bar, ptr %s1, i32 0, i32 1
  %1 = call ptr @llvm.ptr.annotation.p0.p1(ptr %f2, ptr addrspace(1) @.str.5, ptr addrspace(1) @.str.1, i32 9, ptr addrspace(1) undef)
  store i8 0, ptr %1, align 4, !tbaa !13
  ; CHECK-LLVM: %[[FIELD3:.*]] = getelementptr inbounds %struct.bar, ptr %{{[a-zA-Z0-9]+}}, i32 0, i32 2
  ; CHECK-LLVM: call ptr @llvm.ptr.annotation.p0{{.*}}%[[FIELD3]]{{.*}}[[STR6]]
  %f3 = getelementptr inbounds %struct.bar, ptr %s1, i32 0, i32 2
  %2 = call ptr @llvm.ptr.annotation.p0.p1(ptr %f3, ptr addrspace(1) @.str.6, ptr addrspace(1) @.str.1, i32 9, ptr addrspace(1) undef)
  store float 0.000000e+00,ptr %2, align 4, !tbaa !14
  ; CHECK-LLVM: %[[FIELD4:.*]] = getelementptr inbounds %struct.bar, ptr %{{[a-zA-Z0-9]+}}, i32 0, i32 3
  ; CHECK-LLVM: call ptr @llvm.ptr.annotation.p0{{.*}}%[[FIELD4]]{{.*}}[[STR8]]
  %f4 = getelementptr inbounds %struct.bar, ptr %s1, i32 0, i32 3
  %3 = call ptr @llvm.ptr.annotation.p0.p1(ptr %f4, ptr addrspace(1) @.str.7, ptr addrspace(1) @.str.1, i32 9, ptr addrspace(1) @.args.1)
  store i8 0, ptr %3, align 4, !tbaa !13
  call void @llvm.lifetime.end.p0(i64 12, ptr %s1) #4
  ret void
}

; Function Attrs: mustprogress noinline norecurse optnone uwtable
define dso_local noundef i32 @with_zeroinitializer() #0 {
entry:
  %retval = alloca i32, align 4
  %s = alloca %struct.S, align 4
  store i32 0, ptr %retval, align 4
  %a = getelementptr inbounds %struct.S, ptr %s, i32 0, i32 0
  %0 = call ptr @llvm.ptr.annotation.p0.p1(ptr %a, ptr addrspace(1) @.str.8, ptr addrspace(1) @.str.1, i32 3, ptr addrspace(1) @.args.2)
  ; CHECK-LLVM: %[[FIELD5:.*]] = getelementptr inbounds %struct.S, ptr %{{[a-z]+}}, i32 0, i32 0
  ; CHECK-LLVM: call ptr @llvm.ptr.annotation.p0.p0{{.*}}%[[FIELD5]]{{.*}}[[STR9]]
  %1 = load i32, ptr %0, align 4
  call void @_Z3fooi(i32 noundef %1)
  ret i32 0
}

; Function Attrs: mustprogress noinline norecurse optnone uwtable
define dso_local noundef i32 @with_false() #0 {
entry:
  %retval = alloca i32, align 4
  %s = alloca %struct.S.1, align 4
  store i32 0, ptr %retval, align 4
  %a = getelementptr inbounds %struct.S.1, ptr %s, i32 0, i32 0
  %0 = call ptr @llvm.ptr.annotation.p0.p1(ptr %a, ptr addrspace(1) @.str.9, ptr addrspace(1) @.str.1, i32 3, ptr addrspace(1) @.args.3)
  ; CHECK-LLVM: %[[FIELD6:.*]] = getelementptr inbounds %struct.S.1, ptr %{{[a-z]+}}, i32 0, i32 0
  ; CHECK-LLVM: call ptr @llvm.ptr.annotation.p0{{.*}}%[[FIELD6]]{{.*}}[[STR10]]
  %1 = load i32, ptr %0, align 4
  call void @_Z3fooi(i32 noundef %1)
  ret i32 0
}

; Function Attrs: mustprogress noinline norecurse optnone uwtable
define dso_local noundef i32 @mixed() #0 {
entry:
  %retval = alloca i32, align 4
  %s = alloca %struct.S.2, align 4
  store i32 0, ptr %retval, align 4
  %a = getelementptr inbounds %struct.S.2, ptr %s, i32 0, i32 0
  %0 = call ptr @llvm.ptr.annotation.p0.p1(ptr %a, ptr addrspace(1) @.str.10, ptr addrspace(1) @.str.1, i32 3, ptr addrspace(1) @.args.4)
  ; CHECK-LLVM: %[[FIELD7:.*]] = getelementptr inbounds %struct.S.2, ptr %{{[a-z]+}}, i32 0, i32 0
  ; CHECK-LLVM: call ptr @llvm.ptr.annotation.p0{{.*}}%[[FIELD7]]{{.*}}[[STR11]]
  %1 = load i32, ptr %0, align 4
  call void @_Z3fooi(i32 noundef %1)
  ret i32 0
}

; Function Attrs: mustprogress norecurse
define weak_odr dso_local spir_kernel void @_ZTSZ11TestKernelAvE4MyIP(ptr addrspace(1) noundef align 4 %0) local_unnamed_addr #5 !kernel_arg_buffer_location !15 !sycl_kernel_omit_args !16 {
  %2 = alloca %struct.MyIP, align 8
  call void @llvm.lifetime.start.p0(i64 8, ptr nonnull %2) #4
  %3 = addrspacecast ptr %2 to ptr addrspace(4)
  %4 = call ptr addrspace(4) @llvm.ptr.annotation.p4.p1(ptr addrspace(4) %3, ptr addrspace(1) @.str.11, ptr addrspace(1) @.str.1.12, i32 13, ptr addrspace(1) @.args)
  ; CHECK-LLVM: %[[ALLOCA:.*]] = alloca %struct.MyIP, align 8
  ; CHECK-LLVM: %[[ASCAST:.*]] = addrspacecast ptr %[[ALLOCA]] to ptr addrspace(4)
  ; CHECK-LLVM: call void @llvm.var.annotation{{.*}}(ptr addrspace(4) %[[ASCAST]], ptr [[STR12]], ptr undef, i32 undef, ptr undef)
  %5 = addrspacecast ptr addrspace(1) %0 to ptr addrspace(4)
  store ptr addrspace(4) %5, ptr addrspace(4) %4, align 8, !tbaa !17
  %6 = load ptr addrspace(4), ptr addrspace(4) %4, align 8, !tbaa !17
  %7 = load i32, ptr addrspace(4) %6, align 4, !tbaa !19
  %8 = shl nsw i32 %7, 1
  store i32 %8, ptr addrspace(4) %6, align 4, !tbaa !19
  call void @llvm.lifetime.end.p0(i64 8, ptr nonnull %2) #4
  ret void
}

; Function Attrs: mustprogress noinline norecurse optnone uwtable
define dso_local noundef i32 @with_true() #0 {
entry:
  %retval = alloca i32, align 4
  %s = alloca %struct.S.3, align 4
  store i32 0, ptr %retval, align 4
  %a = getelementptr inbounds %struct.S.3, ptr %s, i32 0, i32 0
  %0 = call ptr @llvm.ptr.annotation.p0.p1(ptr %a, ptr addrspace(1) @.str.12, ptr addrspace(1) @.str.1, i32 3, ptr addrspace(1) @.args.5)
  ; CHECK-LLVM: %[[FIELD8:.*]] = getelementptr inbounds %struct.S.3, ptr %{{[a-z]+}}, i32 0, i32 0
  ; CHECK-LLVM: call ptr @llvm.ptr.annotation.p0{{.*}}%[[FIELD8]]{{.*}}[[STR13]]
  %1 = load i32, ptr %0, align 4
  call void @_Z3fooi(i32 noundef %1)
  ret i32 0
}

declare dso_local void @_Z3fooi(i32 noundef)

; Function Attrs: nounwind
declare ptr @llvm.ptr.annotation.p0.p1(ptr, ptr addrspace(1), ptr addrspace(1), i32, ptr addrspace(1)) #4

; Function Attrs: nounwind

; Function Attrs: nounwind
declare ptr addrspace(4) @llvm.ptr.annotation.p4.p1(ptr addrspace(4), ptr addrspace(1), ptr addrspace(1), i32, ptr addrspace(1)) #4

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
