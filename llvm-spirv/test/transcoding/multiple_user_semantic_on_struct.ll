; This test is intended to check that we are actually using the
; ptr.annotation intrinsic call results during the reverse translation.

; Source (https://godbolt.org/z/qzhsKfPeq):
; class B {
; public:
;     int Val [[clang::annotate("ClassB")]];
; };
; class A {
; public:
;     int Val [[clang::annotate("ClassA")]];
;     int MultiDec [[clang::annotate("Dec1"), clang::annotate("Dec2"), clang::annotate("Dec3")]];
;     [[clang::annotate("ClassAfieldB")]]class B b;
; };
; void foo(int);
; int main() {
;     A a;
;     B b;
;     A c;
;     foo(a.Val);       // ClassA
;     foo(c.Val);       // Obj2ClassA
;     foo(a.MultiDec);  // ClassAMultiDec
;     foo(a.b.Val);     // ClassAFieldB
;     foo(b.Val);       // ClassB
;     return 0;
; }

; RUN: llvm-as %s -o %t.bc
; RUN: llvm-spirv %t.bc -spirv-text -o - | FileCheck %s --check-prefix=CHECK-SPIRV

; RUN: llvm-spirv %t.bc -o %t.spv
; RUN: llvm-spirv -r %t.spv -o %t.rev.bc
; RUN: llvm-dis < %t.rev.bc | FileCheck %s --check-prefix=CHECK-LLVM

; Check that even when FPGA memory extensions are enabled - yet we have
; UserSemantic decoration be generated
; RUN: llvm-as %s -o %t.bc
; RUN: llvm-spirv %t.bc --spirv-ext=+SPV_INTEL_fpga_memory_accesses,+SPV_INTEL_fpga_memory_attributes -spirv-text -o - | FileCheck %s --check-prefix=CHECK-SPIRV

; CHECK-SPIRV: Decorate [[#GEP1:]] UserSemantic "ClassA"
; CHECK-SPIRV: Decorate [[#GEP2:]] UserSemantic "ClassA"
; CHECK-SPIRV: Decorate [[#GEP3:]] UserSemantic "Dec1"
; CHECK-SPIRV: Decorate [[#GEP3]] UserSemantic "Dec2"
; CHECK-SPIRV: Decorate [[#GEP3]] UserSemantic "Dec3"
; CHECK-SPIRV: Decorate [[#GEP4:]] UserSemantic "ClassAfieldB"
; CHECK-SPIRV: Decorate [[#GEP5:]] UserSemantic "ClassB"
; CHECK-SPIRV: Decorate [[#GEP6:]] UserSemantic "ClassB"
; CHECK-SPIRV: InBoundsPtrAccessChain [[#]] [[#GEP1]] [[#]]
; CHECK-SPIRV: InBoundsPtrAccessChain [[#]] [[#GEP2]] [[#]]
; CHECK-SPIRV: InBoundsPtrAccessChain [[#]] [[#GEP3]] [[#]]
; CHECK-SPIRV: InBoundsPtrAccessChain [[#]] [[#GEP4]] [[#]]
; CHECK-SPIRV: InBoundsPtrAccessChain [[#]] [[#GEP5]] [[#]]
; CHECK-SPIRV: InBoundsPtrAccessChain [[#]] [[#GEP6]] [[#]]

target datalayout = "e-i64:64-v16:16-v24:32-v32:32-v48:64-v96:128-v192:256-v256:256-v512:512-v1024:1024-n8:16:32:64"
target triple = "spir64"

%class.A = type { i32, i32, %class.B }
%class.B = type { i32 }

@.str = private unnamed_addr addrspace(1) constant [7 x i8] c"ClassA\00", section "llvm.metadata"
@.str.1 = private unnamed_addr addrspace(1) constant [17 x i8] c"/app/example.cpp\00", section "llvm.metadata"
@.str.2 = private unnamed_addr addrspace(1) constant [5 x i8] c"Dec1\00", section "llvm.metadata"
@.str.3 = private unnamed_addr addrspace(1) constant [5 x i8] c"Dec2\00", section "llvm.metadata"
@.str.4 = private unnamed_addr addrspace(1) constant [5 x i8] c"Dec3\00", section "llvm.metadata"
@.str.5 = private unnamed_addr addrspace(1) constant [13 x i8] c"ClassAfieldB\00", section "llvm.metadata"
@.str.6 = private unnamed_addr addrspace(1) constant [7 x i8] c"ClassB\00", section "llvm.metadata"

; CHECK-LLVM-DAG: @[[#StrStructA:]] = {{.*}}"ClassA\00"
; CHECK-LLVM-DAG: @[[#Dec1:]] = {{.*}}"Dec1\00"
; CHECK-LLVM-DAG: @[[#Dec2:]] = {{.*}}"Dec2\00"
; CHECK-LLVM-DAG: @[[#Dec3:]] = {{.*}}"Dec3\00"
; CHECK-LLVM-DAG: @[[#StrAfieldB:]] = {{.*}}"ClassAfieldB\00"
; CHECK-LLVM-DAG: @[[#StrStructB:]] = {{.*}}"ClassB\00"

define dso_local noundef i32 @main() #0 {
  %1 = alloca i32, align 4
  %2 = alloca %class.A, align 4
  %3 = alloca %class.B, align 4
  %4 = alloca %class.A, align 4
; CHECK-LLVM: %[[#ObjClassA:]] = alloca %class.A, align 4
; CHECK-LLVM: %[[#ObjClassB:]] = alloca %class.B, align 4
; CHECK-LLVM: %[[#Obj2ClassA:]] = alloca %class.A, align 4

  store i32 0, ptr %1, align 4
  %5 = getelementptr inbounds %class.A, ptr %2, i32 0, i32 0
  %6 = call ptr @llvm.ptr.annotation.p0.p1(ptr %5, ptr addrspace(1) @.str, ptr addrspace(1) @.str.1, i32 11, ptr addrspace(1) null)
  ; CHECK-LLVM: %[[#GepClassAVal:]] = getelementptr inbounds %class.A, ptr %[[#ObjClassA]], i32 0, i32 0
  ; CHECK-LLVM: %[[#PtrAnnClassAVal:]] = call ptr @llvm.ptr.annotation{{.*}}(ptr %[[#GepClassAVal]], ptr @[[#StrStructA]]
  %7 = load i32, ptr %6, align 4
 ; CHECK-LLVM: %[[#LoadClassAVal:]] = bitcast ptr %[[#PtrAnnClassAVal]] to ptr
  ; CHECK-LLVM: %[[#LoadClassAVal2:]] = bitcast ptr %[[#LoadClassAVal]] to ptr
  ; CHECK-LLVM: %[[#CallClassA:]] = load i32, ptr %[[#LoadClassAVal2]], align 4
  call void @_Z3fooi(i32 noundef %7)
  %8 = getelementptr inbounds %class.A, ptr %4, i32 0, i32 0
  %9 = call ptr @llvm.ptr.annotation.p0.p1(ptr %8, ptr addrspace(1) @.str, ptr addrspace(1) @.str.1, i32 11, ptr addrspace(1) null)
  ; CHECK-LLVM: %[[#GepObj2ClassA:]] = getelementptr inbounds %class.A, ptr %[[#Obj2ClassA]], i32 0, i32 0
  ; CHECK-LLVM: %[[#PtrAnnObj2ClassA:]] = call ptr @llvm.ptr.annotation{{.*}}(ptr %[[#GepObj2ClassA]], ptr @[[#StrStructA]]
  %10 = load i32, ptr %9, align 4
  ; CHECK-LLVM: %[[#LoadObj2ClassA:]] = bitcast ptr %[[#PtrAnnObj2ClassA]] to ptr
  ; CHECK-LLVM: %[[#LoadObj2ClassA2:]] = bitcast ptr %[[#LoadObj2ClassA]] to ptr
  ; CHECK-LLVM: %[[#CallObj2ClassA:]] = load i32, ptr %[[#LoadObj2ClassA2]], align 4
  call void @_Z3fooi(i32 noundef %10)
  %11 = getelementptr inbounds %class.A, ptr %2, i32 0, i32 1
  %12 = call ptr @llvm.ptr.annotation.p0.p1(ptr %11, ptr addrspace(1) @.str.2, ptr addrspace(1) @.str.1, i32 12, ptr addrspace(1) null)
  %13 = call ptr @llvm.ptr.annotation.p0.p1(ptr %12, ptr addrspace(1) @.str.3, ptr addrspace(1) @.str.1, i32 12, ptr addrspace(1) null)
  %14 = call ptr @llvm.ptr.annotation.p0.p1(ptr %13, ptr addrspace(1) @.str.4, ptr addrspace(1) @.str.1, i32 12, ptr addrspace(1) null)
  ; CHECK-LLVM: %[[#GepMultiDec:]] = getelementptr inbounds %class.A, ptr %[[#ObjClassA]], i32 0, i32 1
  ; CHECK-LLVM: %[[#PtrAnnMultiDec:]] = call ptr @llvm.ptr.annotation{{.*}}(ptr %[[#GepMultiDec]], ptr @[[#Dec1]]
  ; CHECK-LLVM: %[[#PtrAnn2MultiDec:]] = call ptr @llvm.ptr.annotation{{.*}}(ptr %[[#PtrAnnMultiDec]], ptr @[[#Dec2]]
  ; CHECK-LLVM: %[[#PtrAnn3MultiDec:]] = call ptr @llvm.ptr.annotation{{.*}}(ptr %[[#PtrAnnMultiDec]], ptr @[[#Dec3]]
  %15 = load i32, ptr %14, align 4
  ; CHECK-LLVM: %[[#LoadMultiDec:]] = bitcast ptr %[[#PtrAnnMultiDec]] to ptr
  ; CHECK-LLVM: %[[#LoadMultiDec2:]] = bitcast ptr %[[#LoadMultiDec]] to ptr
  ; CHECK-LLVM: %[[#CallClassAMultiDec:]] = load i32, ptr %[[#LoadMultiDec2]], align 4
  call void @_Z3fooi(i32 noundef %15)
  %16 = getelementptr inbounds %class.A, ptr %2, i32 0, i32 2
  %17 = call ptr @llvm.ptr.annotation.p0.p1(ptr %16, ptr addrspace(1) @.str.5, ptr addrspace(1) @.str.1, i32 13, ptr addrspace(1) null)
  ; CHECK-LLVM: %[[#GepClassAFieldB:]] = getelementptr inbounds %class.A, ptr %[[#ObjClassA]], i32 0, i32 2
  ; CHECK-LLVM: %[[#PtrAnnClassAFieldB:]] = call ptr @llvm.ptr.annotation{{.*}}(ptr %[[#GepClassAFieldB]], ptr @[[#StrAfieldB]]
  %18 = getelementptr inbounds %class.B, ptr %17, i32 0, i32 0
  %19 = call ptr @llvm.ptr.annotation.p0.p1(ptr %18, ptr addrspace(1) @.str.6, ptr addrspace(1) @.str.1, i32 5, ptr addrspace(1) null)
  ; CHECK-LLVM: %[[#CastGepClassAFieldB:]] = bitcast ptr %[[#GepClassAFieldB]] to ptr
  ; CHECK-LLVM: %[[#CastGepClassAFieldB2:]] = bitcast ptr %[[#CastGepClassAFieldB]] to ptr
  ; CHECK-LLVM: %[[#GEPClassB:]] = getelementptr inbounds %class.B, ptr %[[#CastGepClassAFieldB2]], i32 0, i32 0
  ; CHECK-LLVM: %[[#PtrAnnClassB:]] = call ptr @llvm.ptr.annotation{{.*}}(ptr %[[#GEPClassB]], ptr @[[#StrStructB]]
  %20 = load i32, ptr %19, align 4
  ; CHECK-LLVM: %[[#Cast2ClassB:]] = bitcast ptr %[[#PtrAnnClassB]] to ptr
  ; CHECK-LLVM: %[[#Cast2ClassB2:]] = bitcast ptr %[[#Cast2ClassB]] to ptr
  ; CHECK-LLVM: %[[#CallClassAFieldB:]] = load i32, ptr %[[#Cast2ClassB2]], align 4
  call void @_Z3fooi(i32 noundef %20)
  %21 = getelementptr inbounds %class.B, ptr %3, i32 0, i32 0
  %22 = call ptr @llvm.ptr.annotation.p0.p1(ptr %21, ptr addrspace(1) @.str.6, ptr addrspace(1) @.str.1, i32 5, ptr addrspace(1) null)
  ; CHECK-LLVM: %[[#GEP2ClassB:]] = getelementptr inbounds %class.B, ptr %[[#ObjClassB]], i32 0, i32 0
  ; CHECK-LLVM: %[[#PtrAnn2ClassB:]] = call ptr @llvm.ptr.annotation{{.*}}(ptr %[[#GEP2ClassB]], ptr @[[#StrStructB]]
  %23 = load i32, ptr %22, align 4
  ; CHECK-LLVM: %[[#Cast2ClassB:]] = bitcast ptr %[[#PtrAnn2ClassB]] to ptr
  ; CHECK-LLVM: %[[#Cast2ClassB2:]] = bitcast ptr %[[#Cast2ClassB]] to ptr
  ; CHECK-LLVM: %[[#CallClassB:]] = load i32, ptr %[[#Cast2ClassB2]], align 4
  call void @_Z3fooi(i32 noundef %23)
  ret i32 0
}

declare void @_Z3fooi(i32 noundef) #2

declare ptr @llvm.ptr.annotation.p0.p1(ptr, ptr addrspace(1), ptr addrspace(1), i32, ptr addrspace(1)) #3

attributes #0 = { mustprogress noinline norecurse optnone uwtable "frame-pointer"="all" "min-legal-vector-width"="0" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "tune-cpu"="generic" }
attributes #1 = { nocallback nofree nosync nounwind speculatable willreturn memory(none) }
attributes #2 = { "frame-pointer"="all" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "tune-cpu"="generic" }
attributes #3 = { nocallback nofree nosync nounwind willreturn memory(inaccessiblemem: readwrite) }
